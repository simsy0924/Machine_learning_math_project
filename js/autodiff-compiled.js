// Compiled reverse-mode plan for ordinary pure loss expressions.
//
// The math and the update order are unchanged; this only removes interpreter
// overhead. The topological order, the per-node input arrays and the list of
// 변수 steps are computed once per graph structure and then reused for every
// SGD step, and adjoints live in an array indexed by node id instead of a Map.

(function installCompiledGradientStrategy() {
  let planVersion = '';
  const plans = new Map();

  function compileGradientPlan(outputId, structureCache) {
    const topo = cachedTopoForOutput(structureCache, outputId);
    const steps = [];
    const variableSteps = [];

    for (const id of topo) {
      const node = graph.nodes.get(id);
      if (!node) throw new Error('존재하지 않는 블록입니다.');
      const def = getBlockDef(node.type);

      // Stateful/special execution inside a loss expression needs the
      // interpreted evaluator; leave those graphs alone.
      if (def.special === 'repeat' || def.special === 'setVariable' || def.special === 'derivative') {
        return null;
      }

      const inputIds = new Array(def.inputs.length);
      for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
        const connection = cachedGraphInput(structureCache, id, inputIndex);
        if (!connection) throw new Error(`입력 '${def.inputs[inputIndex]}'이 연결되지 않았습니다.`);
        inputIds[inputIndex] = connection.from;
      }

      const step = { id, node, def, inputIds, inputs: new Array(inputIds.length) };
      steps.push(step);
      if (node.type === 'variable') variableSteps.push(step);
    }

    return {
      outputId,
      steps,
      variableSteps,
      values: [],
      adjoints: [],
      touchedAdjoints: []
    };
  }

  function planFor(outputId, structureCache) {
    const version = graphStructureVersion(structureCache);
    if (planVersion !== version) {
      planVersion = version;
      plans.clear();
    }

    if (plans.has(outputId)) return plans.get(outputId);
    const plan = compileGradientPlan(outputId, structureCache);
    plans.set(outputId, plan);
    return plan;
  }

  function executeGradientPlan(plan) {
    const values = plan.values;
    const adjoints = plan.adjoints;
    const touched = plan.touchedAdjoints;

    for (let i = 0; i < touched.length; i++) adjoints[touched[i]] = undefined;
    touched.length = 0;

    // Forward pass. Input arrays are allocated once when the plan is compiled.
    for (const step of plan.steps) {
      const inputs = step.inputs;
      for (let i = 0; i < step.inputIds.length; i++) inputs[i] = values[step.inputIds[i]];
      values[step.id] = step.def.compute(step.node, inputs);
    }

    const output = values[plan.outputId];
    if (typeof output !== 'number') throw new Error('미분 블록의 식 출력은 숫자 하나여야 합니다.');

    adjoints[plan.outputId] = 1;
    touched.push(plan.outputId);

    for (let k = plan.steps.length - 1; k >= 0; k--) {
      const step = plan.steps[k];
      const upstream = adjoints[step.id];
      if (upstream == null || !step.inputIds.length) continue;

      const inputs = step.inputs;
      const grads = step.node.type.startsWith('custom:')
        ? userBlockVJP(USER_BLOCKS.get(step.node.type.slice(7)), inputs, upstream)
        : primitiveVJP(step.node.type, inputs, values[step.id], upstream);

      for (let i = 0; i < grads.length; i++) {
        const gradient = grads[i];
        if (gradient == null) continue;
        const inputId = step.inputIds[i];
        const previous = adjoints[inputId];
        if (previous == null) {
          adjoints[inputId] = gradient;
          touched.push(inputId);
        } else {
          adjoints[inputId] = addValues(previous, gradient);
        }
      }
    }

    const gradients = new Map();
    for (const step of plan.variableSteps) {
      const name = String(step.node.params.name || 'x');
      const gradient = adjoints[step.id] ?? zerosLike(values[step.id]);
      const previous = gradients.get(name);
      gradients.set(name, previous == null ? gradient : addValues(previous, gradient));
    }
    return gradients;
  }

  registerGradientStrategy({
    name: 'compiled',
    priority: 20,
    plan: planFor,
    execute: executeGradientPlan
  });
})();
