// Fast executor for user-authored manual-backprop definitions.
//
// The mathematical definition is unchanged. This file only changes how an
// immutable saved backward graph is executed:
//   - compile the required gradient outputs once,
//   - skip gradient branches whose destination name is blank,
//   - reuse matching internal values from an already executed custom forward,
//   - reuse scratch arrays and cached gradient-variable lookup.
//
// Reusing a matching forward subexpression is ordinary common-subexpression
// elimination. No derivative is inferred and no backward formula is generated.

const MANUAL_BACKPROP_EXECUTION_PLANS = new WeakMap();
const MANUAL_BACKPROP_VARIABLE_NODE_CACHE = new Map();

function manualBackpropVariableNodeCached(name) {
  const key = String(name || '');
  const cached = MANUAL_BACKPROP_VARIABLE_NODE_CACHE.get(key);
  if (
    cached &&
    graph.nodes.get(cached.id) === cached &&
    cached.type === 'variable' &&
    String(cached.params.name || 'x') === key
  ) {
    return cached;
  }

  const node = findVariableNode(key);
  if (!node) throw new Error(`'${key}'이라는 변수 블록을 찾지 못했습니다.`);
  MANUAL_BACKPROP_VARIABLE_NODE_CACHE.set(key, node);
  return node;
}

function readManualBackpropRuntimeVariable(name) {
  return readRuntimeVariable(manualBackpropVariableNodeCached(name));
}

function writeManualBackpropRuntimeVariable(name, value) {
  const key = String(name || '');
  const variableNode = manualBackpropVariableNodeCached(key);
  // Manual gradient scratch values have to be visible to the very next backward
  // block in the same iteration, so this intentionally writes immediately.
  RUNTIME_VARIABLES.set(key, {
    value,
    signature: variableSignature(variableNode)
  });
  return value;
}

function manualBackpropSourceExpressionKey(sourceId, inputCount, flags) {
  if (sourceId === MANUAL_BACKPROP_UPSTREAM_ID) return 'G';
  if (sourceId === MANUAL_BACKPROP_OUTPUT_ID) {
    flags.usesForwardOutput = true;
    return 'Y';
  }
  if (sourceId < 0) {
    const index = -sourceId - 1;
    if (index >= 0 && index < inputCount) return `I:${index}`;
  }
  throw new Error('역전파 정의의 입력 소스를 찾지 못했습니다.');
}

function compileManualBackpropExecutionPlan(
  definition,
  inputCount,
  requiredMask,
  forwardDefinition = null
) {
  if (!definition || !Array.isArray(definition.nodes) || !Array.isArray(definition.connections)) {
    throw new Error('이 블록의 역전파 정의가 올바르지 않습니다.');
  }

  const outputIds = Array.isArray(definition.gradientOutputNodeIds)
    ? definition.gradientOutputNodeIds.map(Number)
    : [];
  if (outputIds.length !== inputCount || outputIds.some(id => !Number.isFinite(id))) {
    throw new Error('모든 입력의 기울기 출력이 지정되지 않았습니다.');
  }

  const nodes = new Map();
  for (const saved of definition.nodes) {
    const id = Number(saved.id);
    if (!Number.isInteger(id) || id <= 0 || nodes.has(id)) {
      throw new Error('역전파 정의의 계산 블록 ID가 올바르지 않습니다.');
    }
    nodes.set(id, sanitizedManualNode(saved));
  }

  // Preserve the old evaluator's first-connection-wins behaviour for malformed
  // imported files.
  const connectionByInput = new Map();
  for (const connection of definition.connections) {
    const to = Number(connection.to);
    const inputIndex = Number(connection.inputIndex);
    const key = `${to}:${inputIndex}`;
    if (!connectionByInput.has(key)) connectionByInput.set(key, Number(connection.from));
  }

  const forwardExpressionMap = forwardDefinition
    ? userBlockForwardExpressionMap(forwardDefinition)
    : null;
  const flags = { usesForwardOutput: false, usesForwardTrace: false };
  const infoByNodeId = new Map();
  const expressionVisiting = new Set();

  function expressionFor(id) {
    const numericId = Number(id);
    if (numericId < 0) return manualBackpropSourceExpressionKey(numericId, inputCount, flags);

    const existing = infoByNodeId.get(numericId);
    if (existing) return existing.expressionKey;
    if (expressionVisiting.has(numericId)) throw new Error('역전파 정의에 순환 연결이 있습니다.');

    const node = nodes.get(numericId);
    if (!node) throw new Error('역전파 정의의 계산 블록을 찾지 못했습니다.');
    if (!manualBackpropInternalBlockAllowed(node.type)) {
      let title = node.type;
      try { title = getBlockDef(node.type).title; } catch { /* keep raw type */ }
      throw new Error(`'${title}' 블록은 역전파 정의 안에서 사용할 수 없습니다.`);
    }

    const def = getBlockDef(node.type);
    const sourceIds = new Array(def.inputs.length);
    const inputExpressionKeys = new Array(def.inputs.length);

    expressionVisiting.add(numericId);
    try {
      for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
        const from = connectionByInput.get(`${numericId}:${inputIndex}`);
        if (from == null) {
          throw new Error(`역전파 정의에서 '${def.inputs[inputIndex]}' 입력이 비어 있습니다.`);
        }
        sourceIds[inputIndex] = from;
        inputExpressionKeys[inputIndex] = expressionFor(from);
      }
    } finally {
      expressionVisiting.delete(numericId);
    }

    const expressionKey = userBlockExpressionKeyFor(node.type, node.params, inputExpressionKeys);
    infoByNodeId.set(numericId, { def, sourceIds, expressionKey });
    return expressionKey;
  }

  const order = [];
  const visited = new Set();
  const visiting = new Set();
  const cachedForwardIndexByNodeId = new Map();

  function visitForExecution(id) {
    const numericId = Number(id);
    if (numericId < 0) {
      manualBackpropSourceExpressionKey(numericId, inputCount, flags);
      return;
    }
    if (visited.has(numericId)) return;
    if (visiting.has(numericId)) throw new Error('역전파 정의에 순환 연결이 있습니다.');

    const expressionKey = expressionFor(numericId);
    if (forwardExpressionMap?.has(expressionKey)) {
      cachedForwardIndexByNodeId.set(numericId, forwardExpressionMap.get(expressionKey));
      flags.usesForwardTrace = true;
      visited.add(numericId);
      order.push(numericId);
      return;
    }

    const { sourceIds } = infoByNodeId.get(numericId);
    visiting.add(numericId);
    try {
      for (const sourceId of sourceIds) visitForExecution(sourceId);
    } finally {
      visiting.delete(numericId);
    }
    visited.add(numericId);
    order.push(numericId);
  }

  for (let i = 0; i < inputCount; i++) {
    if (requiredMask[i]) visitForExecution(outputIds[i]);
  }

  const valueIndexByNodeId = new Map();
  for (let i = 0; i < order.length; i++) valueIndexByNodeId.set(order[i], i);

  const steps = new Array(order.length);
  for (let valueIndex = 0; valueIndex < order.length; valueIndex++) {
    const id = order[valueIndex];
    const node = nodes.get(id);
    const cachedForwardValueIndex = cachedForwardIndexByNodeId.get(id);

    if (cachedForwardValueIndex != null) {
      steps[valueIndex] = {
        valueIndex,
        node,
        kind: 'forwardCache',
        forwardValueIndex: cachedForwardValueIndex,
        def: null,
        refs: [],
        inputs: []
      };
      continue;
    }

    const { def, sourceIds } = infoByNodeId.get(id);
    const refs = sourceIds.map(sourceId => {
      if (sourceId < 0) return sourceId;
      const ref = valueIndexByNodeId.get(sourceId);
      if (ref == null) throw new Error('역전파 정의 내부 연결을 찾지 못했습니다.');
      return ref;
    });

    steps[valueIndex] = {
      valueIndex,
      node,
      kind: 'compute',
      forwardValueIndex: null,
      def,
      refs,
      inputs: new Array(refs.length)
    };
  }

  const outputRefs = new Array(inputCount).fill(null);
  for (let i = 0; i < inputCount; i++) {
    if (!requiredMask[i]) continue;
    const outputId = outputIds[i];
    if (outputId < 0) {
      manualBackpropSourceExpressionKey(outputId, inputCount, flags);
      outputRefs[i] = outputId;
      continue;
    }
    const ref = valueIndexByNodeId.get(outputId);
    if (ref == null) throw new Error('역전파 정의의 기울기 출력을 찾지 못했습니다.');
    outputRefs[i] = ref;
  }

  return {
    definition,
    inputCount,
    requiredMask: [...requiredMask],
    usesForwardOutput: flags.usesForwardOutput,
    usesForwardTrace: flags.usesForwardTrace,
    steps,
    outputRefs,
    values: new Array(steps.length),
    gradients: new Array(inputCount)
  };
}

function manualBackpropExecutionPlanFor(
  definition,
  inputCount,
  requiredMask = null,
  forwardDefinition = null
) {
  const mask = requiredMask || new Array(inputCount).fill(true);
  const maskKey = mask.map(Boolean).map(value => value ? '1' : '0').join('');
  const ownerKey = forwardDefinition ? String(forwardDefinition.id || 'custom') : 'builtin';
  const key = `${inputCount}|${maskKey}|${ownerKey}`;

  let plans = MANUAL_BACKPROP_EXECUTION_PLANS.get(definition);
  if (!plans) {
    plans = new Map();
    MANUAL_BACKPROP_EXECUTION_PLANS.set(definition, plans);
  }

  let plan = plans.get(key);
  if (!plan) {
    plan = compileManualBackpropExecutionPlan(definition, inputCount, mask, forwardDefinition);
    plans.set(key, plan);
  }
  return plan;
}

function readManualBackpropPlanRef(ref, plan, inputValues, forwardOutput, upstream) {
  if (ref >= 0) return plan.values[ref];
  if (ref === MANUAL_BACKPROP_UPSTREAM_ID) return upstream;
  if (ref === MANUAL_BACKPROP_OUTPUT_ID) return forwardOutput;

  const inputIndex = -ref - 1;
  if (inputIndex >= 0 && inputIndex < inputValues.length) return inputValues[inputIndex];
  throw new Error('역전파 정의의 입력 소스를 찾지 못했습니다.');
}

function executeCompiledManualBackpropPlan(
  plan,
  inputValues,
  forwardOutput,
  upstream,
  forwardTrace = null
) {
  const values = plan.values;

  for (const step of plan.steps) {
    if (step.kind === 'forwardCache') {
      if (!forwardTrace) throw new Error('사용자 블록 순전파 중간값 캐시가 없습니다.');
      values[step.valueIndex] = forwardTrace.values[step.forwardValueIndex];
      continue;
    }

    const inputs = step.inputs;
    for (let i = 0; i < step.refs.length; i++) {
      inputs[i] = readManualBackpropPlanRef(step.refs[i], plan, inputValues, forwardOutput, upstream);
    }
    values[step.valueIndex] = step.def.compute(step.node, inputs);
  }

  for (let i = 0; i < plan.outputRefs.length; i++) {
    const ref = plan.outputRefs[i];
    plan.gradients[i] = ref == null
      ? undefined
      : readManualBackpropPlanRef(ref, plan, inputValues, forwardOutput, upstream);
  }
  return plan.gradients;
}

// Replace only the execution strategy. The saved formulas, editor, explicit
// gradient variables and user-controlled reverse order are exactly the same as
// manual-backprop.js; no derivative is inferred here.
executeManualBackpropNode = function executeManualBackpropNodeCompiled(type, baseDefinition, node, inputs) {
  const definition = manualBackpropDefinitionForType(type);
  if (!definition) {
    throw new Error(`'${baseDefinition.title}' 블록의 역전파 정의가 없습니다. 먼저 인스펙터에서 직접 정의해 주세요.`);
  }

  const targetNames = new Array(inputs.length);
  const requiredMask = new Array(inputs.length);
  for (let i = 0; i < inputs.length; i++) {
    const targetName = String(node.params[`manualBackpropGradient${i}`] || '').trim();
    targetNames[i] = targetName;
    requiredMask[i] = Boolean(targetName);
  }

  const forwardDefinition = String(type).startsWith('custom:')
    ? USER_BLOCKS.get(String(type).slice(7)) || null
    : null;
  const plan = manualBackpropExecutionPlanFor(
    definition,
    inputs.length,
    requiredMask,
    forwardDefinition
  );

  const upstreamName = String(node.params.manualBackpropUpstream || 'g').trim() || 'g';
  const upstream = readManualBackpropRuntimeVariable(upstreamName);

  let forwardTrace = null;
  if (forwardDefinition && (plan.usesForwardTrace || plan.usesForwardOutput)) {
    forwardTrace = userBlockForwardTraceFor(forwardDefinition, inputs)
      || ensureUserBlockForwardTrace(forwardDefinition, inputs);
  }

  const forwardOutput = plan.usesForwardOutput
    ? (forwardDefinition
        ? userBlockForwardOutputFromTrace(forwardDefinition, forwardTrace)
        : baseDefinition.compute(node, inputs))
    : undefined;

  const gradients = executeCompiledManualBackpropPlan(
    plan,
    inputs,
    forwardOutput,
    upstream,
    forwardTrace
  );

  for (let i = 0; i < gradients.length; i++) {
    if (!targetNames[i]) continue;
    writeManualBackpropRuntimeVariable(targetNames[i], gradients[i]);
  }

  // The ordinary node output is only for sequencing/inspection. If the first
  // gradient was intentionally left unused, returning upstream avoids computing
  // an otherwise dead branch just to create a sequencing value.
  return gradients[0] ?? upstream;
};
