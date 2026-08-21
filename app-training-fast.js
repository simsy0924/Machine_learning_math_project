// Compiled fast path for SGD training.
//
// The math and update order are unchanged. This file removes interpreter overhead
// from the hot path by compiling both the leaf repeat body and pure reverse-mode
// autodiff graphs once, then reusing their execution plans for every SGD step.

(function installCompiledTrainingRuntime() {
  let gradientPlanVersion = '';
  const gradientPlans = new Map();

  function structureVersion(cache) {
    return `${cache.nodeCount}:${cache.connectionCount}:${cache.hashA}:${cache.hashB}`;
  }

  // ---------- Compiled reverse-mode autodiff ----------

  function compileGradientPlan(outputId, structureCache) {
    const topo = cachedTopoForOutput(structureCache, outputId);
    const steps = [];
    const variableSteps = [];

    for (const id of topo) {
      const node = graph.nodes.get(id);
      if (!node) throw new Error('존재하지 않는 블록입니다.');
      const def = getBlockDef(node.type);

      // Preserve the established evaluator for unusual stateful loss graphs.
      if (def.special === 'repeat' || def.special === 'setVariable' || def.special === 'derivative') {
        return null;
      }

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
        inputs: new Array(inputIds.length)
      };
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

  function getGradientPlan(outputId, structureCache) {
    const version = structureVersion(structureCache);
    if (gradientPlanVersion !== version) {
      gradientPlanVersion = version;
      gradientPlans.clear();
    }

    if (gradientPlans.has(outputId)) return gradientPlans.get(outputId);
    const plan = compileGradientPlan(outputId, structureCache);
    gradientPlans.set(outputId, plan);
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

    // Reverse pass using arrays indexed by node id instead of per-step Maps.
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

  const computeAllGraphGradientsBeforeCompiledTraining = computeAllGraphGradients;
  computeAllGraphGradients = function(outputId) {
    const structureCache = ensureGraphStructureCache();
    const plan = getGradientPlan(outputId, structureCache);
    if (!plan) return computeAllGraphGradientsBeforeCompiledTraining(outputId);
    return executeGradientPlan(plan);
  };

  // ---------- Compiled leaf repeat body ----------

  // A derivative block points to the scalar expression that differentiateGraph
  // evaluates itself. Treat that edge as metadata rather than eagerly executing
  // the loss once before immediately executing it again for backpropagation.
  function collectTrainingExecutionOrder(outputId, structureCache) {
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

  function compileTrainingLeafPlan(outputId, structureCache) {
    const order = collectTrainingExecutionOrder(outputId, structureCache);
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
      outputId,
      steps,
      values: [],
      pendingUpdates: new Map(),
      loopFrame: new Map(),
      progressFrame: { name: '', index: 0, count: 0 }
    };
  }

  function readTrainingVariable(step) {
    const loopValue = activeLoopValue(step.variableName);
    if (loopValue.found) return loopValue.value;

    const stored = RUNTIME_VARIABLES.get(step.variableName);
    if (!stored || stored.signature !== step.variableSignature) {
      const value = initialVariableValue(step.node);
      RUNTIME_VARIABLES.set(step.variableName, {
        value,
        signature: step.variableSignature
      });
      return value;
    }
    return stored.value;
  }

  function writeTrainingVariable(step, value) {
    const entry = {
      value,
      signature: step.variableSignature
    };
    if (pendingVariableUpdates) pendingVariableUpdates.set(step.variableName, entry);
    else RUNTIME_VARIABLES.set(step.variableName, entry);
    return value;
  }

  function executeTrainingLeafPlan(plan) {
    const values = plan.values;

    for (const step of plan.steps) {
      let value;
      if (step.kind === 'variable') {
        value = readTrainingVariable(step);
      } else if (step.kind === 'setVariable') {
        value = writeTrainingVariable(step, values[step.inputIds[0]]);
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

  function trainingLeafPlanFor(repeatId, outputId, structureCache, progress) {
    const version = structureVersion(structureCache);
    let state = progress.trainingFastPlans;
    if (!state || state.version !== version) {
      state = { version, plans: new Map() };
      progress.trainingFastPlans = state;
    }

    let plan = state.plans.get(repeatId);
    if (!plan || plan.outputId !== outputId) {
      plan = compileTrainingLeafPlan(outputId, structureCache);
      state.plans.set(repeatId, plan);
    }
    return plan;
  }

  const evaluateRepeatProgressiveBeforeCompiledTraining = evaluateRepeatProgressive;
  evaluateRepeatProgressive = async function(nodeId, memo, visiting, progress) {
    // Selected evaluation already has its own compiled executor and profiler.
    if (typeof SELECTED_FAST_EVAL_DEPTH !== 'undefined' && SELECTED_FAST_EVAL_DEPTH > 0) {
      return evaluateRepeatProgressiveBeforeCompiledTraining(nodeId, memo, visiting, progress);
    }

    if (memo.has(nodeId)) return memo.get(nodeId);
    if (visiting.has(nodeId)) throw new Error('순환 연결은 계산할 수 없습니다.');

    const node = graph.nodes.get(nodeId);
    if (!node) throw new Error('존재하지 않는 블록입니다.');
    const structureCache = progress.graphCache || ensureGraphStructureCache();
    const connection = cachedGraphInput(structureCache, nodeId, 0);
    if (!connection) throw new Error("입력 '실행할 식'이 연결되지 않았습니다.");

    const isLeafRepeat = !branchContainsRepeat(connection.from, new Set(), structureCache);
    if (!isLeafRepeat) {
      return evaluateRepeatProgressiveBeforeCompiledTraining(nodeId, memo, visiting, progress);
    }

    visiting.add(nodeId);
    const count = repeatCount(node);
    const start = repeatStartValue(node);
    const indexName = String(node.params.indexVariable || 'i');
    const leafPlan = trainingLeafPlanFor(nodeId, connection.from, structureCache, progress);
    let last = 0;
    let lastIndex = null;

    try {
      for (let offset = 0; offset < count; offset++) {
        const loopValue = start + offset;
        lastIndex = loopValue;
        const parentPendingUpdates = pendingVariableUpdates;

        const loopFrame = leafPlan.loopFrame;
        loopFrame.clear();
        loopFrame.set(indexName, loopValue);

        const progressFrame = leafPlan.progressFrame;
        progressFrame.name = `${indexName}=${loopValue}`;
        progressFrame.index = offset;
        progressFrame.count = count;

        leafPlan.pendingUpdates.clear();
        pendingVariableUpdates = leafPlan.pendingUpdates;
        LOOP_CONTEXT_STACK.push(loopFrame);
        progress.stack.push(progressFrame);

        try {
          last = executeTrainingLeafPlan(leafPlan);
          commitPendingVariableUpdates();
          progress.completed++;

          // Avoid an async Promise/await round-trip on every single SGD step.
          // Check the clock only every eight steps and yield at the same ~80 ms
          // UI cadence used by the normal progressive evaluator.
          if ((progress.completed & 7) === 0 && performance.now() - progress.lastYield >= 80) {
            await maybeYieldProgress(progress);
          }
        } finally {
          pendingVariableUpdates = parentPendingUpdates;
          progress.stack.pop();
          LOOP_CONTEXT_STACK.pop();
        }
      }

      if (lastIndex != null && findVariableNode(indexName)) {
        writeRuntimeVariable(indexName, lastIndex, true);
      }

      memo.set(nodeId, last);
      return last;
    } finally {
      visiting.delete(nodeId);
    }
  };
})();
