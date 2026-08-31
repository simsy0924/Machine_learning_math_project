// Forward-only compiled execution for 사용자 블록 (user-created blocks).
//
// A 사용자 블록 is only a reusable mathematical function. Derivatives are not
// inferred from its internal graph. Manual backward definitions live in
// js/manual-backprop.js and are authored explicitly by the user.

const USER_BLOCK_PLANS = new WeakMap();

function userBlockSlotKey(nodeId, inputIndex) {
  return `${nodeId}:${inputIndex == null ? 'source' : inputIndex}`;
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
  for (let valueIndex = 0; valueIndex < order.length; valueIndex++) {
    const nodeId = order[valueIndex];
    const node = nodeById.get(nodeId);
    const sourceSlot = externalIndexBySlot.get(userBlockSlotKey(nodeId, null));

    if (sourceSlot != null) {
      steps[valueIndex] = {
        valueIndex,
        node,
        kind: 'externalSource',
        sourceSlot,
        inputRefs: [],
        inputs: []
      };
      continue;
    }

    const blockDef = getBlockDef(node.type);
    if (blockDef.special) {
      throw new Error(`'${blockDef.title}' 블록은 사용자 블록 내부에서 실행할 수 없습니다.`);
    }

    const inputRefs = new Array(blockDef.inputs.length);
    for (let inputIndex = 0; inputIndex < blockDef.inputs.length; inputIndex++) {
      const key = userBlockSlotKey(nodeId, inputIndex);
      const fromNodeId = connectionByInput.get(key);
      if (fromNodeId != null) {
        const internalIndex = valueIndexByNodeId.get(fromNodeId);
        if (internalIndex == null) throw new Error('사용자 블록 내부 연결을 찾지 못했습니다.');
        inputRefs[inputIndex] = internalIndex;
      } else {
        const externalIndex = externalIndexBySlot.get(key);
        if (externalIndex == null) throw new Error(`사용자 블록 입력 '${blockDef.inputs[inputIndex]}'이 비어 있습니다.`);
        // Negative values encode external input slots: -1 -> slot 0.
        inputRefs[inputIndex] = -externalIndex - 1;
      }
    }

    steps[valueIndex] = {
      valueIndex,
      node,
      kind: 'compute',
      blockDef,
      inputRefs,
      inputs: new Array(inputRefs.length)
    };
  }

  const outputValueIndex = valueIndexByNodeId.get(definition.outputNodeId);
  if (outputValueIndex == null) throw new Error('사용자 블록 출력 노드를 찾지 못했습니다.');

  return {
    definition,
    steps,
    outputValueIndex,
    scratchValues: new Array(steps.length)
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

function readUserBlockRef(ref, values, externalValues) {
  return ref >= 0 ? values[ref] : externalValues[-ref - 1];
}

function executeUserBlockForward(plan, externalValues) {
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
    values[step.valueIndex] = step.blockDef.compute(step.node, inputs);
  }

  return values[plan.outputValueIndex];
}

function evaluateUserDefinition(definition, externalValues) {
  return executeUserBlockForward(userBlockPlanFor(definition), externalValues);
}
