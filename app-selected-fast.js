// Compiled fast path for selected bulk evaluation.
// Async work remains only at repeat/yield boundaries. A leaf repeat body is
// compiled once into a topological execution plan and then run synchronously,
// avoiding recursive calls, Map memoization, graph lookups, and per-node input
// array allocation on every test sample.

let SELECTED_FAST_EVAL_DEPTH = 0;

function compileSelectedLeafPlan(outputId, structureCache = ensureGraphStructureCache()) {
  const topo = cachedTopoForOutput(structureCache, outputId);
  const steps = [];

  for (const id of topo) {
    const node = graph.nodes.get(id);
    if (!node) throw new Error('존재하지 않는 블록입니다.');
    const def = getBlockDef(node.type);

    if (def.special === 'repeat') {
      throw new Error('빠른 반복 계산 내부에서 중첩 반복을 직접 실행할 수 없습니다.');
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

function readCompiledVariable(step) {
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

function writeCompiledVariable(step, value) {
  const entry = {
    value,
    signature: step.variableSignature
  };

  if (pendingVariableUpdates) {
    pendingVariableUpdates.set(step.variableName, entry);
  } else {
    RUNTIME_VARIABLES.set(step.variableName, entry);
  }

  return value;
}

function executeSelectedLeafPlan(plan) {
  const values = plan.values;

  for (const step of plan.steps) {
    let value;

    if (step.kind === 'variable') {
      value = readCompiledVariable(step);
    } else if (step.kind === 'setVariable') {
      value = writeCompiledVariable(step, values[step.inputIds[0]]);
    } else if (step.kind === 'derivative') {
      const expressionId = step.inputIds[0];
      value = differentiateGraph(expressionId, String(step.node.params.variable || 'x'));
    } else {
      const inputs = step.inputs;
      for (let i = 0; i < step.inputIds.length; i++) {
        inputs[i] = values[step.inputIds[i]];
      }
      value = step.def.compute(step.node, inputs);
    }

    values[step.id] = value;
  }

  return values[plan.outputId];
}

// Activate the compiled leaf executor only while the user is running "선택 계산".
// Full training remains untouched.
const evaluateSelectedStatefulBranchBeforeCompiledLeaf = evaluateSelectedStatefulBranch;
evaluateSelectedStatefulBranch = async function(nodeId) {
  SELECTED_FAST_EVAL_DEPTH++;
  try {
    return await evaluateSelectedStatefulBranchBeforeCompiledLeaf(nodeId);
  } finally {
    SELECTED_FAST_EVAL_DEPTH--;
  }
};

const evaluateRepeatProgressiveBeforeCompiledSelected = evaluateRepeatProgressive;
evaluateRepeatProgressive = async function(nodeId, memo, visiting, progress) {
  if (SELECTED_FAST_EVAL_DEPTH <= 0) {
    return evaluateRepeatProgressiveBeforeCompiledSelected(nodeId, memo, visiting, progress);
  }

  if (memo.has(nodeId)) return memo.get(nodeId);
  if (visiting.has(nodeId)) throw new Error('순환 연결은 계산할 수 없습니다.');

  const node = graph.nodes.get(nodeId);
  if (!node) throw new Error('존재하지 않는 블록입니다.');
  const structureCache = progress.graphCache || ensureGraphStructureCache();
  const connection = cachedGraphInput(structureCache, nodeId, 0);
  if (!connection) throw new Error("입력 '실행할 식'이 연결되지 않았습니다.");

  visiting.add(nodeId);
  const count = repeatCount(node);
  const start = repeatStartValue(node);
  const indexName = String(node.params.indexVariable || 'i');
  const isLeafRepeat = !branchContainsRepeat(connection.from, new Set(), structureCache);
  const leafPlan = isLeafRepeat ? compileSelectedLeafPlan(connection.from, structureCache) : null;
  let last = 0;
  let lastIndex = null;

  try {
    for (let offset = 0; offset < count; offset++) {
      const loopValue = start + offset;
      lastIndex = loopValue;
      const parentPendingUpdates = pendingVariableUpdates;

      let loopFrame;
      let progressFrame;

      if (leafPlan) {
        loopFrame = leafPlan.loopFrame;
        loopFrame.clear();
        loopFrame.set(indexName, loopValue);

        progressFrame = leafPlan.progressFrame;
        progressFrame.name = `${indexName}=${loopValue}`;
        progressFrame.index = offset;
        progressFrame.count = count;

        leafPlan.pendingUpdates.clear();
        pendingVariableUpdates = leafPlan.pendingUpdates;
      } else {
        loopFrame = new Map([[indexName, loopValue]]);
        progressFrame = {
          name: `${indexName}=${loopValue}`,
          index: offset,
          count
        };
        pendingVariableUpdates = new Map();
      }

      LOOP_CONTEXT_STACK.push(loopFrame);
      progress.stack.push(progressFrame);

      try {
        if (leafPlan) {
          last = executeSelectedLeafPlan(leafPlan);
        } else {
          last = await evaluateNodeProgressive(connection.from, new Map(), new Set(), progress);
        }

        commitPendingVariableUpdates();

        if (isLeafRepeat) {
          progress.completed++;
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
