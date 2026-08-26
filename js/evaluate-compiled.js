// Compiled execution plans for the body of a leaf 반복.
//
// A leaf repeat is one whose body contains no further repeat, so its body runs
// unchanged tens of thousands of times. Compiling it once into a flat list of
// steps removes the recursion, the Map memoization, the graph lookups and the
// per-node input array from every iteration.
//
// Two execution orders exist, because 선택 계산 and 학습 need different things
// from a 미분 block:
//
//   'selected' — the plain topological order. Everything the branch reaches is
//                evaluated, which is what a test/inspection run expects.
//   'training' — a 미분 block's expression input is treated as metadata rather
//                than a step, because differentiateGraph runs that pass itself.
//                Without this the loss would be evaluated twice per SGD step.

function selectedLeafOrder(outputId, structureCache) {
  const topo = cachedTopoForOutput(structureCache, outputId);
  for (const id of topo) {
    const node = graph.nodes.get(id);
    if (!node) throw new Error('존재하지 않는 블록입니다.');
    if (getBlockDef(node.type).special === 'repeat') {
      throw new Error('빠른 반복 계산 내부에서 중첩 반복을 직접 실행할 수 없습니다.');
    }
  }
  return topo;
}

function trainingLeafOrder(outputId, structureCache) {
  const seen = new Set();
  const visiting = new Set();
  const order = [];

  const visit = id => {
    if (seen.has(id)) return;
    if (visiting.has(id)) throw new Error('순환 연결은 계산할 수 없습니다.');

    const node = graph.nodes.get(id);
    if (!node) throw new Error('존재하지 않는 블록입니다.');
    const def = getBlockDef(node.type);
    if (def.special === 'repeat') throw new Error('빠른 학습 반복 안에는 중첩 반복을 직접 넣을 수 없습니다.');

    visiting.add(id);
    if (def.special === 'derivative') {
      const connection = cachedGraphInput(structureCache, id, 0);
      if (!connection) throw new Error("입력 '식'이 연결되지 않았습니다.");
      // Do not visit connection.from here: differentiateGraph owns that pass.
    } else {
      for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
        const connection = cachedGraphInput(structureCache, id, inputIndex);
        if (!connection) throw new Error(`입력 '${def.inputs[inputIndex]}'이 연결되지 않았습니다.`);
        visit(connection.from);
      }
    }
    visiting.delete(id);
    seen.add(id);
    order.push(id);
  };

  visit(outputId);
  return order;
}

const LEAF_PLAN_ORDERS = {
  selected: selectedLeafOrder,
  training: trainingLeafOrder
};

function compileLeafPlan(mode, outputId, structureCache) {
  const order = LEAF_PLAN_ORDERS[mode](outputId, structureCache);
  const steps = [];

  for (const id of order) {
    const node = graph.nodes.get(id);
    if (!node) throw new Error('존재하지 않는 블록입니다.');
    const def = getBlockDef(node.type);

    const inputIds = new Array(def.inputs.length);
    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const connection = cachedGraphInput(structureCache, id, inputIndex);
      if (!connection) throw new Error(`입력 '${def.inputs[inputIndex]}'이 연결되지 않았습니다.`);
      inputIds[inputIndex] = connection.from;
    }

    const step = {
      id,
      node,
      def,
      inputIds,
      inputs: new Array(inputIds.length),
      kind: 'normal',
      variableName: null,
      variableSignature: null
    };

    if (node.type === 'variable') {
      step.kind = 'variable';
      step.variableName = String(node.params.name || 'x');
      step.variableSignature = variableSignature(node);
    } else if (def.special === 'setVariable') {
      const variableName = String(node.params.variable || 'w');
      const variableNode = findVariableNode(variableName);
      if (!variableNode) throw new Error(`'${variableName}'이라는 변수 블록을 찾지 못했습니다.`);
      step.kind = 'setVariable';
      step.variableName = variableName;
      step.variableSignature = variableSignature(variableNode);
    } else if (def.special === 'derivative') {
      step.kind = 'derivative';
    }

    steps.push(step);
  }

  return {
    mode,
    outputId,
    steps,
    values: [],
    pendingUpdates: new Map(),
    loopFrame: new Map(),
    progressFrame: { name: '', index: 0, count: 0 },
    arena: createResultArena()
  };
}

// Plans are cached on the in-flight progress object: one Calculate/선택 계산 run
// pins the graph structure, and the cache is dropped when the run ends.
function leafPlanFor(mode, repeatId, outputId, structureCache, progress) {
  const version = graphStructureVersion(structureCache);
  let state = progress.leafPlans;
  if (!state || state.version !== version) {
    state = { version, plans: new Map() };
    progress.leafPlans = state;
  }

  const key = `${mode}:${repeatId}`;
  let plan = state.plans.get(key);
  if (!plan || plan.outputId !== outputId) {
    plan = compileLeafPlan(mode, outputId, structureCache);
    state.plans.set(key, plan);
  }
  return plan;
}

// The compiled equivalents of the 변수 / 값 바꾸기 evaluator branches. The
// variable's signature was resolved when the plan was compiled, so the hot path
// does not rebuild it per step.
function readCompiledVariable(step) {
  const loopValue = activeLoopValue(step.variableName);
  if (loopValue.found) return loopValue.value;

  const stored = RUNTIME_VARIABLES.get(step.variableName);
  if (!stored || stored.signature !== step.variableSignature) {
    const value = initialVariableValue(step.node);
    RUNTIME_VARIABLES.set(step.variableName, { value, signature: step.variableSignature });
    return value;
  }
  return stored.value;
}

function writeCompiledVariable(step, value) {
  const entry = { value, signature: step.variableSignature };
  if (pendingVariableUpdates) pendingVariableUpdates.set(step.variableName, entry);
  else RUNTIME_VARIABLES.set(step.variableName, entry);
  return value;
}

function runLeafPlanSteps(plan) {
  const values = plan.values;

  for (const step of plan.steps) {
    let value;

    if (step.kind === 'variable') {
      value = readCompiledVariable(step);
    } else if (step.kind === 'setVariable') {
      value = writeCompiledVariable(step, values[step.inputIds[0]]);
    } else if (step.kind === 'derivative') {
      value = differentiateGraph(step.inputIds[0], String(step.node.params.variable || 'x'));
    } else {
      const inputs = step.inputs;
      for (let i = 0; i < step.inputIds.length; i++) inputs[i] = values[step.inputIds[i]];
      value = step.def.compute(step.node, inputs);
    }

    values[step.id] = value;
  }

  return values[plan.outputId];
}

function executeLeafPlan(plan) {
  const hook = PROFILER_HOOKS.selectedLeafStep;
  if (!hook) return runLeafPlanSteps(plan);

  const token = hook.begin();
  if (token == null) return runLeafPlanSteps(plan);
  try {
    return runLeafPlanSteps(plan);
  } finally {
    hook.end(token);
  }
}
