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
