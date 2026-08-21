// Fast path for selected bulk evaluation.
// Keep async work only at repeat/yield boundaries; evaluate one leaf iteration
// synchronously with cached graph connections so ordinary math nodes do not pay
// an async/await cost on every edge.

let SELECTED_FAST_EVAL_DEPTH = 0;

function evaluateSelectedLeafSync(nodeId, memo = new Map(), visiting = new Set(), structureCache = ensureGraphStructureCache()) {
  if (memo.has(nodeId)) return memo.get(nodeId);
  if (visiting.has(nodeId)) throw new Error('순환 연결은 계산할 수 없습니다.');

  const node = graph.nodes.get(nodeId);
  if (!node) throw new Error('존재하지 않는 블록입니다.');
  const def = getBlockDef(node.type);

  // A leaf repeat body must not contain another repeat. Nested repeats stay on
  // the progressive async path and their innermost leaf is optimized there.
  if (def.special === 'repeat') {
    throw new Error('빠른 반복 계산 내부에서 중첩 반복을 직접 실행할 수 없습니다.');
  }

  visiting.add(nodeId);
  try {
    if (def.special === 'derivative') {
      const connection = cachedGraphInput(structureCache, nodeId, 0);
      if (!connection) throw new Error("입력 '식'이 연결되지 않았습니다.");
      const value = differentiateGraph(connection.from, String(node.params.variable || 'x'));
      memo.set(nodeId, value);
      return value;
    }

    if (def.special === 'setVariable') {
      const connection = cachedGraphInput(structureCache, nodeId, 0);
      if (!connection) throw new Error("입력 '새 값'이 연결되지 않았습니다.");
      const value = evaluateSelectedLeafSync(connection.from, memo, visiting, structureCache);
      const result = writeRuntimeVariable(String(node.params.variable || 'w'), value);
      memo.set(nodeId, result);
      return result;
    }

    const inputs = new Array(def.inputs.length);
    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const connection = cachedGraphInput(structureCache, nodeId, inputIndex);
      if (!connection) throw new Error(`입력 '${def.inputs[inputIndex]}'이 연결되지 않았습니다.`);
      inputs[inputIndex] = evaluateSelectedLeafSync(connection.from, memo, visiting, structureCache);
    }

    const value = def.compute(node, inputs);
    memo.set(nodeId, value);
    return value;
  } finally {
    visiting.delete(nodeId);
  }
}

// Activate the fast leaf executor only while the user is running "선택 계산".
// Full training keeps its existing runtime unchanged, which limits this patch to
// the bulk-test path that was slow.
const evaluateSelectedStatefulBranchBeforeFastLeaf = evaluateSelectedStatefulBranch;
evaluateSelectedStatefulBranch = async function(nodeId) {
  SELECTED_FAST_EVAL_DEPTH++;
  try {
    return await evaluateSelectedStatefulBranchBeforeFastLeaf(nodeId);
  } finally {
    SELECTED_FAST_EVAL_DEPTH--;
  }
};

const evaluateRepeatProgressiveBeforeFastSelected = evaluateRepeatProgressive;
evaluateRepeatProgressive = async function(nodeId, memo, visiting, progress) {
  if (SELECTED_FAST_EVAL_DEPTH <= 0) {
    return evaluateRepeatProgressiveBeforeFastSelected(nodeId, memo, visiting, progress);
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
  let last = 0;
  let lastIndex = null;

  try {
    for (let offset = 0; offset < count; offset++) {
      const loopValue = start + offset;
      lastIndex = loopValue;
      const loopFrame = new Map([[indexName, loopValue]]);
      const parentPendingUpdates = pendingVariableUpdates;
      const progressFrame = {
        name: `${indexName}=${loopValue}`,
        index: offset,
        count
      };

      LOOP_CONTEXT_STACK.push(loopFrame);
      progress.stack.push(progressFrame);
      pendingVariableUpdates = new Map();
      try {
        if (isLeafRepeat) {
          // This is the hot path: no Promise creation or await per math node.
          // One memo is still fresh per sample, preserving loop-dependent data.
          last = evaluateSelectedLeafSync(connection.from, new Map(), new Set(), structureCache);
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
