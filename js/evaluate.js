// Graph evaluation: one synchronous evaluator, one progressive (async) one.
//
// evaluateNode is the plain recursive evaluator used for a single value.
// evaluateGraph and evaluateSelectedStatefulBranch use the progressive
// evaluator, which yields to the browser about every 80 ms so a long 반복 shows
// live progress instead of freezing the page. Both understand the same three
// special blocks — 반복, 값 바꾸기, 미분 — and produce the same numbers.

// Greater than zero while 선택 계산 is running, which selects the leaf plan the
// compiled repeat executor builds (see js/evaluate-compiled.js).
let SELECTED_FAST_EVAL_DEPTH = 0;

// Nodes below a 반복 that change state. 계산 evaluates every node in the
// workspace, and re-running a 값 바꾸기 branch outside its loop would apply an
// extra update, so these are shown but not evaluated at the top level.
function collectStateNodesUnderRepeat(structureCache = ensureGraphStructureCache()) {
  const skip = new Set();
  const memo = new Map();

  const containsStateChange = id => {
    if (memo.has(id)) return memo.get(id);
    const node = graph.nodes.get(id);
    if (!node) return false;
    const def = getBlockDef(node.type);
    let contains = def.special === 'setVariable';

    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const connection = cachedGraphInput(structureCache, id, inputIndex);
      if (connection && containsStateChange(connection.from)) contains = true;
    }

    memo.set(id, contains);
    if (contains) skip.add(id);
    return contains;
  };

  for (const node of graph.nodes.values()) {
    if (getBlockDef(node.type).special !== 'repeat') continue;
    const connection = cachedGraphInput(structureCache, node.id, 0);
    if (connection) containsStateChange(connection.from);
  }

  return skip;
}

// ---------- synchronous evaluation ----------

function evaluateRepeatNode(nodeId, node, memo, visiting) {
  if (visiting.has(nodeId)) throw new Error('순환 연결은 계산할 수 없습니다.');
  visiting.add(nodeId);

  try {
    const connection = graphInputConnection(nodeId, 0);
    if (!connection) throw new Error("입력 '실행할 식'이 연결되지 않았습니다.");

    const count = repeatCount(node);
    const start = repeatStartValue(node);
    const indexName = String(node.params.indexVariable || 'i');
    let last = 0;
    let lastIndex = null;

    for (let offset = 0; offset < count; offset++) {
      const loopValue = start + offset;
      lastIndex = loopValue;
      const parentPendingUpdates = pendingVariableUpdates;

      LOOP_CONTEXT_STACK.push(new Map([[indexName, loopValue]]));
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
    // It is not used while the repeat is running; loop-context values are.
    if (lastIndex != null && findVariableNode(indexName)) {
      writeRuntimeVariable(indexName, lastIndex, true);
    }

    memo.set(nodeId, last);
    return last;
  } finally {
    visiting.delete(nodeId);
  }
}

function evaluateNode(nodeId, memo = new Map(), visiting = new Set()) {
  if (memo.has(nodeId)) return memo.get(nodeId);
  const node = graph.nodes.get(nodeId);
  if (!node) throw new Error('존재하지 않는 블록입니다.');
  const def = getBlockDef(node.type);

  if (def.special === 'repeat') return evaluateRepeatNode(nodeId, node, memo, visiting);

  if (visiting.has(nodeId)) throw new Error('순환 연결은 계산할 수 없습니다.');
  visiting.add(nodeId);

  try {
    if (def.special === 'derivative') {
      const connection = graphInputConnection(nodeId, 0);
      if (!connection) throw new Error("입력 '식'이 연결되지 않았습니다.");
      const value = differentiateGraph(connection.from, String(node.params.variable || 'x'));
      memo.set(nodeId, value);
      return value;
    }

    if (def.special === 'setVariable') {
      const connection = graphInputConnection(nodeId, 0);
      if (!connection) throw new Error("입력 '새 값'이 연결되지 않았습니다.");
      const value = evaluateNode(connection.from, memo, visiting);
      const result = writeRuntimeVariable(String(node.params.variable || 'w'), value);
      memo.set(nodeId, result);
      return result;
    }

    const inputs = [];
    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const connection = graphInputConnection(nodeId, inputIndex);
      if (!connection) throw new Error(`입력 '${def.inputs[inputIndex]}'이 연결되지 않았습니다.`);
      inputs.push(evaluateNode(connection.from, memo, visiting));
    }

    const value = def.compute(node, inputs);
    memo.set(nodeId, value);
    return value;
  } finally {
    visiting.delete(nodeId);
  }
}

// ---------- how much work a run will be ----------

function branchContainsRepeat(nodeId, seen = new Set(), structureCache = ensureGraphStructureCache()) {
  if (seen.has(nodeId)) return false;
  seen.add(nodeId);
  const node = graph.nodes.get(nodeId);
  if (!node) return false;
  const def = getBlockDef(node.type);
  if (def.special === 'repeat') return true;
  for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
    const connection = cachedGraphInput(structureCache, nodeId, inputIndex);
    if (connection && branchContainsRepeat(connection.from, seen, structureCache)) return true;
  }
  return false;
}

function repeatLeafWork(nodeId, seen = new Set(), structureCache = ensureGraphStructureCache()) {
  if (seen.has(nodeId)) return 0;
  seen.add(nodeId);
  const node = graph.nodes.get(nodeId);
  if (!node) return 0;
  const def = getBlockDef(node.type);

  if (def.special === 'repeat') {
    const connection = cachedGraphInput(structureCache, nodeId, 0);
    const nested = connection ? repeatLeafWork(connection.from, new Set(seen), structureCache) : 0;
    return repeatCount(node) * Math.max(1, nested);
  }

  let total = 0;
  for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
    const connection = cachedGraphInput(structureCache, nodeId, inputIndex);
    if (connection) total += repeatLeafWork(connection.from, new Set(seen), structureCache);
  }
  return total;
}

function estimateTotalRepeatWork(structureCache = ensureGraphStructureCache()) {
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
      const connection = cachedGraphInput(structureCache, nodeId, inputIndex);
      if (connection) markNestedRepeats(connection.from, seen);
    }
  };

  for (const repeatId of repeatIds) {
    const connection = cachedGraphInput(structureCache, repeatId, 0);
    if (connection) markNestedRepeats(connection.from);
  }

  let total = 0;
  for (const repeatId of repeatIds) {
    if (!nestedRepeatIds.has(repeatId)) total += repeatLeafWork(repeatId, new Set(), structureCache);
  }
  return total;
}

function newProgressState(total, graphCache) {
  return {
    total,
    completed: 0,
    stack: [],
    lastYield: performance.now(),
    rateSampleTime: null,
    rateSampleCompleted: 0,
    stepsPerSecond: null,
    graphCache
  };
}

// ---------- progress display ----------

function updateStepRate(progress, now, force = false) {
  if (progress.rateSampleTime == null) {
    progress.rateSampleTime = now;
    progress.rateSampleCompleted = progress.completed;
    progress.stepsPerSecond = null;
    return;
  }

  const elapsedMs = now - progress.rateSampleTime;
  if (!force && elapsedMs < 400) return;

  const completedDelta = progress.completed - progress.rateSampleCompleted;
  if (elapsedMs > 0 && completedDelta > 0) {
    const currentRate = completedDelta * 1000 / elapsedMs;
    progress.stepsPerSecond = progress.stepsPerSecond == null
      ? currentRate
      : progress.stepsPerSecond * 0.65 + currentRate * 0.35;
  }

  progress.rateSampleTime = now;
  progress.rateSampleCompleted = progress.completed;
}

function formatProgressStatus(progress) {
  const done = Math.min(progress.completed, progress.total);
  const percent = progress.total > 0 ? (done / progress.total) * 100 : 100;
  const parts = [
    `반복 ${done.toLocaleString('ko-KR')} / ${progress.total.toLocaleString('ko-KR')}`,
    `${percent.toFixed(1)}%`
  ];

  if (Number.isFinite(progress.stepsPerSecond)) {
    const rate = progress.stepsPerSecond;
    parts.push(`속도 ${rate >= 100 ? Math.round(rate).toLocaleString('ko-KR') : rate.toFixed(1)} step/s`);
  }

  if (progress.stack.length) {
    const path = progress.stack
      .map(frame => `${frame.name} ${frame.index + 1}/${frame.count}`)
      .join(' · ');
    parts.push(path);
  }
  return parts.join(' · ');
}

async function maybeYieldProgress(progress, force = false) {
  if (PROFILER_HOOKS.progress) PROFILER_HOOKS.progress(progress);
  if (!progress.total) return;
  const now = performance.now();
  updateStepRate(progress, now, force);
  if (!force && now - progress.lastYield < 80) return;
  workspaceStatus.textContent = formatProgressStatus(progress);
  // Yield to a new browser task so the updated count is actually painted.
  await new Promise(resolve => setTimeout(resolve, 0));
  progress.lastYield = performance.now();
}

// ---------- progressive evaluation ----------

async function evaluateRepeatProgressive(nodeId, memo, visiting, progress) {
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

  // Only a leaf repeat can be compiled: a body containing another repeat has to
  // go back through the interpreter so the inner loop can yield.
  const isLeafRepeat = !branchContainsRepeat(connection.from, new Set(), structureCache);
  const mode = SELECTED_FAST_EVAL_DEPTH > 0 ? 'selected' : 'training';
  const leafPlan = isLeafRepeat ? leafPlanFor(mode, nodeId, connection.from, structureCache, progress) : null;

  // Hoisted so the arena call does not allocate a closure per iteration.
  const runIteration = leafPlan
    ? () => {
        const value = executeLeafPlan(leafPlan);
        commitPendingVariableUpdates();
        return value;
      }
    : null;

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
        // The plan owns its frames and pending-update map, so a long training
        // run allocates nothing per step.
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
        progressFrame = { name: `${indexName}=${loopValue}`, index: offset, count };
        pendingVariableUpdates = new Map();
      }

      LOOP_CONTEXT_STACK.push(loopFrame);
      progress.stack.push(progressFrame);

      try {
        if (leafPlan) {
          // The arena covers the commit too, so buffers that just became the
          // new weights are seen as still live.
          last = withResultArena(leafPlan.arena, runIteration);
          progress.completed++;
          if (mode === 'training') {
            // Avoid an async Promise/await round-trip on every single SGD step.
            // Check the clock only every eight steps and yield at the same
            // ~80 ms UI cadence used by the interpreted evaluator.
            if ((progress.completed & 7) === 0 && performance.now() - progress.lastYield >= 80) {
              await maybeYieldProgress(progress);
            }
          } else {
            await maybeYieldProgress(progress);
          }
        } else {
          last = await evaluateNodeProgressive(connection.from, new Map(), new Set(), progress);
          commitPendingVariableUpdates();
          if (isLeafRepeat) {
            progress.completed++;
            await maybeYieldProgress(progress);
          }
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
  const structureCache = progress.graphCache || ensureGraphStructureCache();

  if (def.special === 'repeat') {
    return evaluateRepeatProgressive(nodeId, memo, visiting, progress);
  }

  if (visiting.has(nodeId)) throw new Error('순환 연결은 계산할 수 없습니다.');
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
      const value = await evaluateNodeProgressive(connection.from, memo, visiting, progress);
      const result = writeRuntimeVariable(String(node.params.variable || 'w'), value);
      memo.set(nodeId, result);
      return result;
    }

    const inputs = [];
    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const connection = cachedGraphInput(structureCache, nodeId, inputIndex);
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

// ---------- entry points ----------

// The 계산 button: evaluate every block in the workspace.
async function evaluateGraph() {
  if (evaluateBtn.disabled) return;

  const oldButtonText = evaluateBtn.textContent;
  evaluateBtn.disabled = true;
  evaluateBtn.textContent = '계산 중…';
  let graphCache = null;

  try {
    // Freeze one structural execution plan for this calculation. Runtime values
    // and loop indices still change normally; only nodes/connections are reused.
    graphCache = beginGraphStructureRun();

    const memo = new Map();
    const visiting = new Set();
    const skippedStateNodes = collectStateNodesUnderRepeat(graphCache);
    const progress = newProgressState(estimateTotalRepeatWork(graphCache), graphCache);
    let successCount = 0;
    let errorCount = 0;

    for (const node of graph.nodes.values()) {
      node.lastValue = undefined;
      node.lastError = null;
    }

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
    if (graphCache) endGraphStructureRun();
    evaluateBtn.disabled = false;
    evaluateBtn.textContent = oldButtonText;
  }
}

// 선택 계산 on a branch that contains 반복 or 값 바꾸기.
async function evaluateSelectedStatefulBranch(nodeId) {
  const hook = PROFILER_HOOKS.selectedRun;
  const token = hook ? hook.begin() : null;
  let graphCache = null;
  SELECTED_FAST_EVAL_DEPTH++;

  try {
    graphCache = beginGraphStructureRun();
    const progress = newProgressState(repeatLeafWork(nodeId, new Set(), graphCache), graphCache);

    if (progress.total) {
      workspaceStatus.textContent = formatProgressStatus(progress);
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    const value = await evaluateNodeProgressive(nodeId, new Map(), new Set(), progress);
    if (progress.total) await maybeYieldProgress(progress, true);
    const result = { value, progress };
    if (hook) hook.end(token, result);
    return result;
  } finally {
    SELECTED_FAST_EVAL_DEPTH--;
    if (graphCache) endGraphStructureRun();
    if (hook) hook.finish(token);
  }
}
