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
