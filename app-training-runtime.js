// Runtime guard for stateful branches used inside repeat blocks.
// A state-changing branch must execute only from its repeat block, not once earlier
// when the global Calculate button walks through every visible node.

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
// The original training evaluator capped repeat at 10,000; this runtime layer extends
// that limit to 1,000,000 without changing dataset sample wrapping (10,000/class).
const MAX_REPEAT_COUNT = 1_000_000;
const repeatCountControl = BLOCKS.repeat?.controls?.find(control => control.key === 'count');
if (repeatCountControl) repeatCountControl.max = MAX_REPEAT_COUNT;

const evaluateNodeBeforeExtendedRepeat = evaluateNode;
evaluateNode = function(nodeId, memo = new Map(), visiting = new Set()) {
  if (memo.has(nodeId)) return memo.get(nodeId);
  const node = graph.nodes.get(nodeId);
  if (!node) throw new Error('존재하지 않는 블록입니다.');
  const def = getBlockDef(node.type);

  if (def.special !== 'repeat') {
    return evaluateNodeBeforeExtendedRepeat(nodeId, memo, visiting);
  }

  const connection = graph.connections.find(c => c.to === nodeId && c.inputIndex === 0);
  if (!connection) throw new Error("입력 '실행할 식'이 연결되지 않았습니다.");

  const count = Math.max(0, Math.min(MAX_REPEAT_COUNT, Math.floor(Number(node.params.count) || 0)));
  const indexName = String(node.params.indexVariable || 'i');
  let last = 0;

  for (let i = 0; i < count; i++) {
    const indexNode = findVariableNode(indexName);
    if (indexNode) writeRuntimeVariable(indexName, i, true);
    pendingVariableUpdates = new Map();
    try {
      last = evaluateNode(connection.from, new Map(), new Set());
      commitPendingVariableUpdates();
    } finally {
      pendingVariableUpdates = null;
    }
  }

  memo.set(nodeId, last);
  return last;
};
