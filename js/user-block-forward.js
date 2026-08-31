// Forward-only compiled execution for 사용자 블록 (user-created blocks).
//
// A 사용자 블록 is only a reusable mathematical function. Derivatives are not
// inferred from its internal graph. Manual backward definitions live in
// js/manual-backprop.js and are authored explicitly by the user.
//
// One compiled leaf iteration may call the same 사용자 블록 several times. Since
// every internal block is pure, identical internal subexpressions with identical
// input value identities can be reused during that iteration. We also keep a
// small ring of forward traces so a later manual backward call can reuse values
// that the forward graph already computed instead of rebuilding them.

const USER_BLOCK_PLANS = new WeakMap();
const USER_BLOCK_FORWARD_TRACE_CAPACITY = 64;
let USER_BLOCK_EXECUTION_GENERATION = 0;

function beginUserBlockExecutionGeneration() {
  USER_BLOCK_EXECUTION_GENERATION++;
  if (!Number.isSafeInteger(USER_BLOCK_EXECUTION_GENERATION)) USER_BLOCK_EXECUTION_GENERATION = 1;
}

function userBlockSlotKey(nodeId, inputIndex) {
  return `${nodeId}:${inputIndex == null ? 'source' : inputIndex}`;
}

function userBlockStableParamsKey(params) {
  const source = params && typeof params === 'object' ? params : {};
  const keys = Object.keys(source)
    .filter(key => !key.startsWith('manualBackprop'))
    .sort();
  return keys.map(key => `${key}:${JSON.stringify(source[key])}`).join('|');
}

function userBlockExpressionKeyFor(type, params, inputExpressionKeys) {
  return JSON.stringify([
    String(type || ''),
    userBlockStableParamsKey(params),
    inputExpressionKeys
  ]);
}

function compileUserBlockPlan(definition) {
  if (!definition || !Array.isArray(definition.nodes) || !Array.isArray(definition.connections)) {
    throw new Error('사용자 블록 정의가 올바르지 않습니다.');
  }

  const nodeById = new Map();
  for (const node of definition.nodes) nodeById.set(node.id, node);

  const externalIndexBySlot = new Map();
  const externalInputs = Array.isArray(definition.externalInputs) ? definition.externalInputs : [];
  for (let i = 0; i < externalInputs.length; i++) {
    const slot = externalInputs[i];
    const key = userBlockSlotKey(slot.nodeId, slot.inputIndex);
    if (!externalIndexBySlot.has(key)) externalIndexBySlot.set(key, i);
  }

  // Preserve Array.find semantics if a malformed imported block contains
  // duplicate connections: the first matching connection wins.
  const connectionByInput = new Map();
  for (const connection of definition.connections) {
    const key = userBlockSlotKey(connection.to, connection.inputIndex);
    if (!connectionByInput.has(key)) connectionByInput.set(key, connection.from);
  }

  const order = [];
  const seen = new Set();
  const visiting = new Set();

  const visit = nodeId => {
    if (seen.has(nodeId)) return;
    if (visiting.has(nodeId)) throw new Error('사용자 블록 내부에 순환 연결이 있습니다.');

    const node = nodeById.get(nodeId);
    if (!node) throw new Error('사용자 블록 내부 노드를 찾을 수 없습니다.');

    visiting.add(nodeId);
    const sourceSlot = externalIndexBySlot.get(userBlockSlotKey(nodeId, null));

    if (sourceSlot == null) {
      const blockDef = getBlockDef(node.type);
      if (blockDef.special) {
        throw new Error(`'${blockDef.title}' 블록은 순수 함수가 아니므로 사용자 블록 내부에서 실행할 수 없습니다.`);
      }

      for (let inputIndex = 0; inputIndex < blockDef.inputs.length; inputIndex++) {
        const key = userBlockSlotKey(nodeId, inputIndex);
        const fromNodeId = connectionByInput.get(key);
        if (fromNodeId != null) {
          visit(fromNodeId);
        } else if (!externalIndexBySlot.has(key)) {
          throw new Error(`사용자 블록 입력 '${blockDef.inputs[inputIndex]}'이 비어 있습니다.`);
        }
      }
    }

    visiting.delete(nodeId);
    seen.add(nodeId);
    order.push(nodeId);
  };

  visit(definition.outputNodeId);

  const valueIndexByNodeId = new Map();
  for (let i = 0; i < order.length; i++) valueIndexByNodeId.set(order[i], i);

  const steps = new Array(order.length);
  const expressionIndexByKey = new Map();

  for (let valueIndex = 0; valueIndex < order.length; valueIndex++) {
    const nodeId = order[valueIndex];
    const node = nodeById.get(nodeId);
    const sourceSlot = externalIndexBySlot.get(userBlockSlotKey(nodeId, null));

    if (sourceSlot != null) {
      const expressionKey = `I:${sourceSlot}`;
      steps[valueIndex] = {
        valueIndex,
        node,
        kind: 'externalSource',
        sourceSlot,
        inputRefs: [],
        inputs: [],
        expressionKey,
        memoGeneration: -1,
        memoInputs: [],
        memoValue: undefined
      };
      if (!expressionIndexByKey.has(expressionKey)) expressionIndexByKey.set(expressionKey, valueIndex);
      continue;
    }

    const blockDef = getBlockDef(node.type);
    if (blockDef.special) {
      throw new Error(`'${blockDef.title}' 블록은 사용자 블록 내부에서 실행할 수 없습니다.`);
    }

    const inputRefs = new Array(blockDef.inputs.length);
    const inputExpressionKeys = new Array(blockDef.inputs.length);
    for (let inputIndex = 0; inputIndex < blockDef.inputs.length; inputIndex++) {
      const key = userBlockSlotKey(nodeId, inputIndex);
      const fromNodeId = connectionByInput.get(key);
      if (fromNodeId != null) {
        const internalIndex = valueIndexByNodeId.get(fromNodeId);
        if (internalIndex == null) throw new Error('사용자 블록 내부 연결을 찾지 못했습니다.');
        inputRefs[inputIndex] = internalIndex;
        inputExpressionKeys[inputIndex] = steps[internalIndex].expressionKey;
      } else {
        const externalIndex = externalIndexBySlot.get(key);
        if (externalIndex == null) throw new Error(`사용자 블록 입력 '${blockDef.inputs[inputIndex]}'이 비어 있습니다.`);
        // Negative values encode external input slots: -1 -> slot 0.
        inputRefs[inputIndex] = -externalIndex - 1;
        inputExpressionKeys[inputIndex] = `I:${externalIndex}`;
      }
    }

    const expressionKey = userBlockExpressionKeyFor(node.type, node.params, inputExpressionKeys);
    steps[valueIndex] = {
      valueIndex,
      node,
      kind: 'compute',
      blockDef,
      inputRefs,
      inputs: new Array(inputRefs.length),
      expressionKey,
      memoGeneration: -1,
      memoInputs: new Array(inputRefs.length),
      memoValue: undefined
    };
    if (!expressionIndexByKey.has(expressionKey)) expressionIndexByKey.set(expressionKey, valueIndex);
  }

  const outputValueIndex = valueIndexByNodeId.get(definition.outputNodeId);
  if (outputValueIndex == null) throw new Error('사용자 블록 출력 노드를 찾지 못했습니다.');

  return {
    definition,
    steps,
    outputValueIndex,
    expressionIndexByKey,
    scratchValues: new Array(steps.length),
    executionGeneration: -1,
    traceEntries: [],
    traceCursor: 0
  };
}

function userBlockPlanFor(definition) {
  let plan = USER_BLOCK_PLANS.get(definition);
  if (!plan) {
    plan = compileUserBlockPlan(definition);
    USER_BLOCK_PLANS.set(definition, plan);
  }
  return plan;
}

function prepareUserBlockPlanGeneration(plan) {
  if (plan.executionGeneration === USER_BLOCK_EXECUTION_GENERATION) return;
  plan.executionGeneration = USER_BLOCK_EXECUTION_GENERATION;
  plan.traceCursor = 0;
}

function readUserBlockRef(ref, values, externalValues) {
  return ref >= 0 ? values[ref] : externalValues[-ref - 1];
}

function sameUserBlockInputTuple(saved, current) {
  if (!saved || saved.length !== current.length) return false;
  for (let i = 0; i < saved.length; i++) {
    if (saved[i] !== current[i]) return false;
  }
  return true;
}

function stepMemoHit(step, inputs) {
  if (step.memoGeneration !== USER_BLOCK_EXECUTION_GENERATION) return false;
  return sameUserBlockInputTuple(step.memoInputs, inputs);
}

function rememberStepMemo(step, inputs, value) {
  step.memoGeneration = USER_BLOCK_EXECUTION_GENERATION;
  for (let i = 0; i < inputs.length; i++) step.memoInputs[i] = inputs[i];
  step.memoValue = value;
}

function rememberUserBlockForwardTrace(plan, externalValues, values) {
  prepareUserBlockPlanGeneration(plan);

  let entry;
  if (plan.traceEntries.length < USER_BLOCK_FORWARD_TRACE_CAPACITY) {
    entry = {
      generation: USER_BLOCK_EXECUTION_GENERATION,
      externalValues: new Array(externalValues.length),
      values: new Array(values.length)
    };
    plan.traceEntries.push(entry);
    plan.traceCursor = plan.traceEntries.length % USER_BLOCK_FORWARD_TRACE_CAPACITY;
  } else {
    entry = plan.traceEntries[plan.traceCursor];
    plan.traceCursor = (plan.traceCursor + 1) % USER_BLOCK_FORWARD_TRACE_CAPACITY;
    entry.generation = USER_BLOCK_EXECUTION_GENERATION;
    if (entry.externalValues.length !== externalValues.length) entry.externalValues = new Array(externalValues.length);
    if (entry.values.length !== values.length) entry.values = new Array(values.length);
  }

  for (let i = 0; i < externalValues.length; i++) entry.externalValues[i] = externalValues[i];
  for (let i = 0; i < values.length; i++) entry.values[i] = values[i];
  return entry;
}

function userBlockForwardTraceFor(definition, externalValues) {
  const plan = userBlockPlanFor(definition);
  prepareUserBlockPlanGeneration(plan);

  for (let offset = 0; offset < plan.traceEntries.length; offset++) {
    const index = (plan.traceCursor - 1 - offset + plan.traceEntries.length) % plan.traceEntries.length;
    const entry = plan.traceEntries[index];
    if (entry.generation !== USER_BLOCK_EXECUTION_GENERATION) continue;
    if (sameUserBlockInputTuple(entry.externalValues, externalValues)) return entry;
  }
  return null;
}

function ensureUserBlockForwardTrace(definition, externalValues) {
  const existing = userBlockForwardTraceFor(definition, externalValues);
  if (existing) return existing;
  executeUserBlockForward(userBlockPlanFor(definition), externalValues);
  const created = userBlockForwardTraceFor(definition, externalValues);
  if (!created) throw new Error('사용자 블록 순전파 중간값 캐시를 만들지 못했습니다.');
  return created;
}

function userBlockForwardExpressionMap(definition) {
  return userBlockPlanFor(definition).expressionIndexByKey;
}

function userBlockForwardOutputFromTrace(definition, trace) {
  const plan = userBlockPlanFor(definition);
  return trace.values[plan.outputValueIndex];
}

function executeUserBlockForward(plan, externalValues) {
  prepareUserBlockPlanGeneration(plan);
  const values = plan.scratchValues;

  for (const step of plan.steps) {
    if (step.kind === 'externalSource') {
      values[step.valueIndex] = externalValues[step.sourceSlot];
      continue;
    }

    const inputs = step.inputs;
    for (let i = 0; i < step.inputRefs.length; i++) {
      inputs[i] = readUserBlockRef(step.inputRefs[i], values, externalValues);
    }

    if (stepMemoHit(step, inputs)) {
      values[step.valueIndex] = step.memoValue;
      continue;
    }

    const value = step.blockDef.compute(step.node, inputs);
    rememberStepMemo(step, inputs, value);
    values[step.valueIndex] = value;
  }

  rememberUserBlockForwardTrace(plan, externalValues, values);
  return values[plan.outputValueIndex];
}

function evaluateUserDefinition(definition, externalValues) {
  return executeUserBlockForward(userBlockPlanFor(definition), externalValues);
}
