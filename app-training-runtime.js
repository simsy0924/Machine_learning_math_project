// Runtime guard and execution context for stateful repeat blocks.
// Repeats must own their loop indices and pending variable updates so nested loops
// cannot overwrite one another or leak stale values into dataset calculations.

collectStateNodesUnderRepeat = function() {
  const skip = new Set();
  const memo = new Map();

  const containsStateChange = id => {
    if (memo.has(id)) return memo.get(id);
    const node = graph.nodes.get(id);
    if (!node) return false;
    const def = getBlockDef(node.type);
    let contains = def.special === 'setVariable';

    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const connection = graph.connections.find(c => c.to === id && c.inputIndex === inputIndex);
      if (connection && containsStateChange(connection.from)) contains = true;
    }

    memo.set(id, contains);
    if (contains) skip.add(id);
    return contains;
  };

  for (const node of graph.nodes.values()) {
    if (getBlockDef(node.type).special !== 'repeat') continue;
    const connection = graph.connections.find(c => c.to === node.id && c.inputIndex === 0);
    if (connection) containsStateChange(connection.from);
  }

  return skip;
};

// Keep the UI permissive for large experiments while still having a finite safety cap.
const MAX_REPEAT_COUNT = 1_000_000;
const repeatCountControl = BLOCKS.repeat?.controls?.find(control => control.key === 'count');
if (repeatCountControl) repeatCountControl.max = MAX_REPEAT_COUNT;

// Loop indices are execution-local values, not trainable/runtime variables.
// A stack lets an inner repeat use `i` while an outer repeat still exposes `epoch`.
const LOOP_CONTEXT_STACK = [];

function activeLoopValue(name) {
  const key = String(name);
  for (let i = LOOP_CONTEXT_STACK.length - 1; i >= 0; i--) {
    const frame = LOOP_CONTEXT_STACK[i];
    if (frame.has(key)) return { found: true, value: frame.get(key) };
  }
  return { found: false, value: undefined };
}

const variableComputeBeforeLoopContext = BLOCKS.variable.compute;
BLOCKS.variable.compute = function(node, inputs) {
  const loopValue = activeLoopValue(node.params.name || 'x');
  if (loopValue.found) return loopValue.value;
  return variableComputeBeforeLoopContext(node, inputs);
};

const evaluateNodeBeforeExtendedRepeat = evaluateNode;
evaluateNode = function(nodeId, memo = new Map(), visiting = new Set()) {
  if (memo.has(nodeId)) return memo.get(nodeId);
  const node = graph.nodes.get(nodeId);
  if (!node) throw new Error('존재하지 않는 블록입니다.');
  const def = getBlockDef(node.type);

  if (def.special !== 'repeat') {
    return evaluateNodeBeforeExtendedRepeat(nodeId, memo, visiting);
  }

  if (visiting.has(nodeId)) throw new Error('순환 연결은 계산할 수 없습니다.');
  visiting.add(nodeId);

  const connection = graph.connections.find(c => c.to === nodeId && c.inputIndex === 0);
  if (!connection) {
    visiting.delete(nodeId);
    throw new Error("입력 '실행할 식'이 연결되지 않았습니다.");
  }

  const count = Math.max(0, Math.min(MAX_REPEAT_COUNT, Math.floor(Number(node.params.count) || 0)));
  const indexName = String(node.params.indexVariable || 'i');
  let last = 0;
  let lastIndex = null;

  try {
    for (let i = 0; i < count; i++) {
      lastIndex = i;
      const frame = new Map([[indexName, i]]);
      const parentPendingUpdates = pendingVariableUpdates;

      LOOP_CONTEXT_STACK.push(frame);
      pendingVariableUpdates = new Map();
      try {
        // Every iteration gets a fresh memo so the loop index and dataset sample
        // are recomputed instead of reusing values from the previous iteration.
        last = evaluateNode(connection.from, new Map(), new Set());
        commitPendingVariableUpdates();
      } finally {
        pendingVariableUpdates = parentPendingUpdates;
        LOOP_CONTEXT_STACK.pop();
      }
    }

    // Preserve the final index only for inspection after the repeat ends.
    // It is not used while the repeat is running; loop-context values above are.
    if (lastIndex != null && findVariableNode(indexName)) {
      writeRuntimeVariable(indexName, lastIndex, true);
    }

    memo.set(nodeId, last);
    return last;
  } finally {
    visiting.delete(nodeId);
  }
};

// ---------- Progressive calculation UI ----------
// Long training runs used to block the browser until every repeat finished. The
// progressive evaluator yields to the browser about every 80 ms so the workspace
// status can show a live repeat count without changing the math or update order.

function repeatCount(node) {
  return Math.max(0, Math.min(MAX_REPEAT_COUNT, Math.floor(Number(node?.params?.count) || 0)));
}

function branchContainsRepeat(nodeId, seen = new Set()) {
  if (seen.has(nodeId)) return false;
  seen.add(nodeId);
  const node = graph.nodes.get(nodeId);
  if (!node) return false;
  const def = getBlockDef(node.type);
  if (def.special === 'repeat') return true;
  for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
    const connection = graph.connections.find(c => c.to === nodeId && c.inputIndex === inputIndex);
    if (connection && branchContainsRepeat(connection.from, seen)) return true;
  }
  return false;
}

function repeatLeafWork(nodeId, seen = new Set()) {
  if (seen.has(nodeId)) return 0;
  seen.add(nodeId);
  const node = graph.nodes.get(nodeId);
  if (!node) return 0;
  const def = getBlockDef(node.type);

  if (def.special === 'repeat') {
    const connection = graph.connections.find(c => c.to === nodeId && c.inputIndex === 0);
    const nested = connection ? repeatLeafWork(connection.from, new Set(seen)) : 0;
    return repeatCount(node) * Math.max(1, nested);
  }

  let total = 0;
  for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
    const connection = graph.connections.find(c => c.to === nodeId && c.inputIndex === inputIndex);
    if (connection) total += repeatLeafWork(connection.from, new Set(seen));
  }
  return total;
}

function estimateTotalRepeatWork() {
  const repeatIds = [];
  for (const node of graph.nodes.values()) {
    if (getBlockDef(node.type).special === 'repeat') repeatIds.push(node.id);
  }
  if (!repeatIds.length) return 0;

  const nestedRepeatIds = new Set();
  const markNestedRepeats = (nodeId, seen = new Set()) => {
    if (seen.has(nodeId)) return;
    seen.add(nodeId);
    const node = graph.nodes.get(nodeId);
    if (!node) return;
    const def = getBlockDef(node.type);
    if (def.special === 'repeat') nestedRepeatIds.add(nodeId);
    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const connection = graph.connections.find(c => c.to === nodeId && c.inputIndex === inputIndex);
      if (connection) markNestedRepeats(connection.from, seen);
    }
  };

  for (const repeatId of repeatIds) {
    const connection = graph.connections.find(c => c.to === repeatId && c.inputIndex === 0);
    if (connection) markNestedRepeats(connection.from);
  }

  let total = 0;
  for (const repeatId of repeatIds) {
    if (!nestedRepeatIds.has(repeatId)) total += repeatLeafWork(repeatId);
  }
  return total;
}

function formatProgressStatus(progress) {
  const done = Math.min(progress.completed, progress.total);
  const percent = progress.total > 0 ? (done / progress.total) * 100 : 100;
  const parts = [
    `반복 ${done.toLocaleString('ko-KR')} / ${progress.total.toLocaleString('ko-KR')}`,
    `${percent.toFixed(1)}%`
  ];
  if (progress.stack.length) {
    const path = progress.stack
      .map(frame => `${frame.name} ${frame.index + 1}/${frame.count}`)
      .join(' · ');
    parts.push(path);
  }
  return parts.join(' · ');
}

async function maybeYieldProgress(progress, force = false) {
  if (!progress.total) return;
  const now = performance.now();
  if (!force && now - progress.lastYield < 80) return;
  workspaceStatus.textContent = formatProgressStatus(progress);
  // Yield to a new browser task so the updated count is actually painted.
  await new Promise(resolve => setTimeout(resolve, 0));
  progress.lastYield = performance.now();
}

async function evaluateRepeatProgressive(nodeId, memo, visiting, progress) {
  if (memo.has(nodeId)) return memo.get(nodeId);
  if (visiting.has(nodeId)) throw new Error('순환 연결은 계산할 수 없습니다.');

  const node = graph.nodes.get(nodeId);
  if (!node) throw new Error('존재하지 않는 블록입니다.');
  const connection = graph.connections.find(c => c.to === nodeId && c.inputIndex === 0);
  if (!connection) throw new Error("입력 '실행할 식'이 연결되지 않았습니다.");

  visiting.add(nodeId);
  const count = repeatCount(node);
  const indexName = String(node.params.indexVariable || 'i');
  const isLeafRepeat = !branchContainsRepeat(connection.from);
  let last = 0;
  let lastIndex = null;

  try {
    for (let i = 0; i < count; i++) {
      lastIndex = i;
      const loopFrame = new Map([[indexName, i]]);
      const parentPendingUpdates = pendingVariableUpdates;
      const progressFrame = { name: indexName, index: i, count };

      LOOP_CONTEXT_STACK.push(loopFrame);
      progress.stack.push(progressFrame);
      pendingVariableUpdates = new Map();
      try {
        last = await evaluateNodeProgressive(connection.from, new Map(), new Set(), progress);
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
}

async function evaluateNodeProgressive(nodeId, memo = new Map(), visiting = new Set(), progress) {
  if (memo.has(nodeId)) return memo.get(nodeId);
  const node = graph.nodes.get(nodeId);
  if (!node) throw new Error('존재하지 않는 블록입니다.');
  const def = getBlockDef(node.type);

  if (def.special === 'repeat') {
    return evaluateRepeatProgressive(nodeId, memo, visiting, progress);
  }

  if (visiting.has(nodeId)) throw new Error('순환 연결은 계산할 수 없습니다.');
  visiting.add(nodeId);

  try {
    if (def.special === 'derivative') {
      const connection = graph.connections.find(c => c.to === nodeId && c.inputIndex === 0);
      if (!connection) throw new Error("입력 '식'이 연결되지 않았습니다.");
      const value = differentiateGraph(connection.from, String(node.params.variable || 'x'));
      memo.set(nodeId, value);
      return value;
    }

    if (def.special === 'setVariable') {
      const connection = graph.connections.find(c => c.to === nodeId && c.inputIndex === 0);
      if (!connection) throw new Error("입력 '새 값'이 연결되지 않았습니다.");
      const value = await evaluateNodeProgressive(connection.from, memo, visiting, progress);
      const result = writeRuntimeVariable(String(node.params.variable || 'w'), value);
      memo.set(nodeId, result);
      return result;
    }

    const inputs = [];
    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const connection = graph.connections.find(c => c.to === nodeId && c.inputIndex === inputIndex);
      if (!connection) throw new Error(`입력 '${def.inputs[inputIndex]}'이 연결되지 않았습니다.`);
      inputs.push(await evaluateNodeProgressive(connection.from, memo, visiting, progress));
    }

    const value = def.compute(node, inputs);
    memo.set(nodeId, value);
    return value;
  } finally {
    visiting.delete(nodeId);
  }
}

// Replace the Calculate action with a progressive version. app-boot.js is loaded
// after this file, so its click listener receives this async function directly.
evaluateGraph = async function() {
  if (evaluateBtn.disabled) return;

  const oldButtonText = evaluateBtn.textContent;
  evaluateBtn.disabled = true;
  evaluateBtn.textContent = '계산 중…';

  const memo = new Map();
  const visiting = new Set();
  const skippedStateNodes = collectStateNodesUnderRepeat();
  const progress = {
    total: estimateTotalRepeatWork(),
    completed: 0,
    stack: [],
    lastYield: performance.now()
  };
  let successCount = 0;
  let errorCount = 0;

  for (const node of graph.nodes.values()) {
    node.lastValue = undefined;
    node.lastError = null;
  }

  try {
    if (progress.total) {
      workspaceStatus.textContent = formatProgressStatus(progress);
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    for (const node of graph.nodes.values()) {
      if (skippedStateNodes.has(node.id)) {
        updateNodePreview(node);
        continue;
      }
      try {
        node.lastValue = await evaluateNodeProgressive(node.id, memo, visiting, progress);
        successCount++;
      } catch (error) {
        node.lastError = error.message;
        errorCount++;
      }
      updateNodePreview(node);
    }

    renderInspector();
    if (progress.total) {
      await maybeYieldProgress(progress, true);
      const progressText = formatProgressStatus(progress);
      workspaceStatus.textContent = errorCount
        ? `계산 종료 · ${progressText} · 오류 ${errorCount}`
        : `계산 완료 · ${progressText}`;
    } else {
      workspaceStatus.textContent = errorCount
        ? `${successCount}/${graph.nodes.size} 계산 · 오류 ${errorCount}`
        : `${successCount}/${graph.nodes.size} 계산`;
    }
  } finally {
    evaluateBtn.disabled = false;
    evaluateBtn.textContent = oldButtonText;
  }
};
