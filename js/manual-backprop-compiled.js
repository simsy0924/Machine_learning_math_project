// Fast executor for user-authored manual-backprop definitions.
//
// The mathematical definition is unchanged. This file only compiles an immutable
// saved backward graph once, reuses scratch arrays on every training step, skips
// recomputing the owner's forward output when the authored formula never reads
// the `y` source, and caches named gradient-variable node lookup.
//
// It is loaded after manual-backprop.js so the editor/persistence code remains
// the single source of truth for what a backward definition means.

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

function compileManualBackpropExecutionPlan(definition, inputCount) {
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

  const visited = new Set();
  const visiting = new Set();
  const order = [];
  const inputSourcesByNodeId = new Map();
  let usesForwardOutput = false;

  function validateSource(sourceId) {
    if (sourceId === MANUAL_BACKPROP_UPSTREAM_ID) return;
    if (sourceId === MANUAL_BACKPROP_OUTPUT_ID) {
      usesForwardOutput = true;
      return;
    }
    if (sourceId < 0) {
      const index = -sourceId - 1;
      if (index >= 0 && index < inputCount) return;
    }
    throw new Error('역전파 정의의 입력 소스를 찾지 못했습니다.');
  }

  function visit(id) {
    const numericId = Number(id);
    if (numericId < 0) {
      validateSource(numericId);
      return;
    }
    if (visited.has(numericId)) return;
    if (visiting.has(numericId)) throw new Error('역전파 정의에 순환 연결이 있습니다.');

    const node = nodes.get(numericId);
    if (!node) throw new Error('역전파 정의의 계산 블록을 찾지 못했습니다.');
    if (!manualBackpropInternalBlockAllowed(node.type)) {
      let title = node.type;
      try { title = getBlockDef(node.type).title; } catch { /* keep raw type */ }
      throw new Error(`'${title}' 블록은 역전파 정의 안에서 사용할 수 없습니다.`);
    }

    const def = getBlockDef(node.type);
    const sourceIds = new Array(def.inputs.length);

    visiting.add(numericId);
    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const from = connectionByInput.get(`${numericId}:${inputIndex}`);
      if (from == null) {
        visiting.delete(numericId);
        throw new Error(`역전파 정의에서 '${def.inputs[inputIndex]}' 입력이 비어 있습니다.`);
      }
      sourceIds[inputIndex] = from;
      visit(from);
    }
    visiting.delete(numericId);

    inputSourcesByNodeId.set(numericId, { def, sourceIds });
    visited.add(numericId);
    order.push(numericId);
  }

  for (const outputId of outputIds) visit(outputId);

  const valueIndexByNodeId = new Map();
  for (let i = 0; i < order.length; i++) valueIndexByNodeId.set(order[i], i);

  const steps = new Array(order.length);
  for (let valueIndex = 0; valueIndex < order.length; valueIndex++) {
    const id = order[valueIndex];
    const node = nodes.get(id);
    const { def, sourceIds } = inputSourcesByNodeId.get(id);
    const refs = sourceIds.map(sourceId => {
      if (sourceId < 0) return sourceId;
      const ref = valueIndexByNodeId.get(sourceId);
      if (ref == null) throw new Error('역전파 정의 내부 연결을 찾지 못했습니다.');
      return ref;
    });

    steps[valueIndex] = {
      valueIndex,
      node,
      def,
      refs,
      inputs: new Array(refs.length)
    };
  }

  const outputRefs = outputIds.map(outputId => {
    if (outputId < 0) {
      validateSource(outputId);
      return outputId;
    }
    const ref = valueIndexByNodeId.get(outputId);
    if (ref == null) throw new Error('역전파 정의의 기울기 출력을 찾지 못했습니다.');
    return ref;
  });

  return {
    definition,
    inputCount,
    usesForwardOutput,
    steps,
    outputRefs,
    values: new Array(steps.length),
    gradients: new Array(outputRefs.length)
  };
}

function manualBackpropExecutionPlanFor(definition, inputCount) {
  let plan = MANUAL_BACKPROP_EXECUTION_PLANS.get(definition);
  if (!plan || plan.inputCount !== inputCount) {
    plan = compileManualBackpropExecutionPlan(definition, inputCount);
    MANUAL_BACKPROP_EXECUTION_PLANS.set(definition, plan);
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

function executeCompiledManualBackpropPlan(plan, inputValues, forwardOutput, upstream) {
  const values = plan.values;

  for (const step of plan.steps) {
    const inputs = step.inputs;
    for (let i = 0; i < step.refs.length; i++) {
      inputs[i] = readManualBackpropPlanRef(step.refs[i], plan, inputValues, forwardOutput, upstream);
    }
    values[step.valueIndex] = step.def.compute(step.node, inputs);
  }

  for (let i = 0; i < plan.outputRefs.length; i++) {
    plan.gradients[i] = readManualBackpropPlanRef(
      plan.outputRefs[i],
      plan,
      inputValues,
      forwardOutput,
      upstream
    );
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

  const plan = manualBackpropExecutionPlanFor(definition, inputs.length);
  const upstreamName = String(node.params.manualBackpropUpstream || 'g').trim() || 'g';
  const upstream = readManualBackpropRuntimeVariable(upstreamName);

  // `y` is an optional source in the authored formula. Most useful backward
  // definitions derive what they need directly from the original inputs. In that
  // common case recomputing a large grouped forward graph here was pure waste.
  const forwardOutput = plan.usesForwardOutput
    ? baseDefinition.compute(node, inputs)
    : undefined;

  const gradients = executeCompiledManualBackpropPlan(
    plan,
    inputs,
    forwardOutput,
    upstream
  );

  for (let i = 0; i < gradients.length; i++) {
    const targetName = String(node.params[`manualBackpropGradient${i}`] || '').trim();
    if (!targetName) continue;
    writeManualBackpropRuntimeVariable(targetName, gradients[i]);
  }

  return gradients[0] ?? upstream;
};
