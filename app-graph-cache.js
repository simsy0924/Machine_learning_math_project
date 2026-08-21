// Cache immutable graph structure during long training runs.
// The block graph usually stays unchanged for tens of thousands of SGD steps,
// so repeatedly scanning graph.connections for every input is unnecessary.

const GRAPH_STRUCTURE_CACHE = {
  nodeCount: -1,
  connectionCount: -1,
  hashA: 0,
  hashB: 0,
  inputsByNode: new Map(),
  topoByOutput: new Map(),
  pinDepth: 0
};

function mixGraphHash(hash, value, prime) {
  hash ^= Number(value) | 0;
  return Math.imul(hash, prime) >>> 0;
}

function mixGraphText(hash, text, prime) {
  const value = String(text);
  for (let i = 0; i < value.length; i++) hash = mixGraphHash(hash, value.charCodeAt(i), prime);
  return hash;
}

function graphStructureFingerprint() {
  let hashA = 2166136261 >>> 0;
  let hashB = 2246822519 >>> 0;

  for (const node of graph.nodes.values()) {
    const def = getBlockDef(node.type);
    hashA = mixGraphHash(hashA, node.id, 16777619);
    hashA = mixGraphText(hashA, node.type, 16777619);
    hashA = mixGraphHash(hashA, def.inputs.length, 16777619);

    hashB = mixGraphHash(hashB, node.id * 31 + def.inputs.length, 3266489917);
    hashB = mixGraphText(hashB, node.type, 3266489917);
  }

  for (const connection of graph.connections) {
    hashA = mixGraphHash(hashA, connection.from, 16777619);
    hashA = mixGraphHash(hashA, connection.to, 16777619);
    hashA = mixGraphHash(hashA, connection.inputIndex, 16777619);

    hashB = mixGraphHash(hashB, connection.to, 3266489917);
    hashB = mixGraphHash(hashB, connection.inputIndex, 3266489917);
    hashB = mixGraphHash(hashB, connection.from, 3266489917);
  }

  return {
    nodeCount: graph.nodes.size,
    connectionCount: graph.connections.length,
    hashA,
    hashB
  };
}

function rebuildGraphStructureCache(cache, fingerprint) {
  const inputsByNode = new Map();
  for (const connection of graph.connections) {
    let inputs = inputsByNode.get(connection.to);
    if (!inputs) {
      inputs = [];
      inputsByNode.set(connection.to, inputs);
    }
    inputs[connection.inputIndex] = connection;
  }

  cache.nodeCount = fingerprint.nodeCount;
  cache.connectionCount = fingerprint.connectionCount;
  cache.hashA = fingerprint.hashA;
  cache.hashB = fingerprint.hashB;
  cache.inputsByNode = inputsByNode;
  cache.topoByOutput = new Map();
  return cache;
}

function ensureGraphStructureCache() {
  const cache = GRAPH_STRUCTURE_CACHE;

  // During one Calculate run the evaluator treats the graph structure as an
  // immutable execution plan. This avoids re-hashing every node/connection for
  // every SGD step. After the run is unpinned, the next access fingerprints the
  // graph again and rebuilds if the user edited it.
  if (cache.pinDepth > 0 && cache.nodeCount >= 0) return cache;

  const fingerprint = graphStructureFingerprint();
  if (
    cache.nodeCount === fingerprint.nodeCount &&
    cache.connectionCount === fingerprint.connectionCount &&
    cache.hashA === fingerprint.hashA &&
    cache.hashB === fingerprint.hashB
  ) {
    return cache;
  }

  return rebuildGraphStructureCache(cache, fingerprint);
}

function beginGraphStructureRun() {
  const cache = ensureGraphStructureCache();
  cache.pinDepth++;
  return cache;
}

function endGraphStructureRun() {
  if (GRAPH_STRUCTURE_CACHE.pinDepth > 0) GRAPH_STRUCTURE_CACHE.pinDepth--;
}

function cachedGraphInput(cache, nodeId, inputIndex) {
  return cache.inputsByNode.get(nodeId)?.[inputIndex] || null;
}

function graphInputConnection(nodeId, inputIndex, cache = null) {
  return cachedGraphInput(cache || ensureGraphStructureCache(), nodeId, inputIndex);
}

function cachedTopoForOutput(cache, outputId) {
  const previous = cache.topoByOutput.get(outputId);
  if (previous) return previous;

  const seen = new Set();
  const visiting = new Set();
  const order = [];

  const visit = id => {
    if (seen.has(id)) return;
    if (visiting.has(id)) throw new Error('순환 연결은 계산할 수 없습니다.');
    const node = graph.nodes.get(id);
    if (!node) throw new Error('존재하지 않는 블록입니다.');

    visiting.add(id);
    const def = getBlockDef(node.type);
    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const connection = cachedGraphInput(cache, id, inputIndex);
      if (connection) visit(connection.from);
    }
    visiting.delete(id);
    seen.add(id);
    order.push(id);
  };

  visit(outputId);
  cache.topoByOutput.set(outputId, order);
  return order;
}

// Keep the public helper compatible while reusing the cached topological order.
collectTopo = function(outputId) {
  const cache = ensureGraphStructureCache();
  return cachedTopoForOutput(cache, outputId);
};

const computeAllGraphGradientsBeforeStructureCache = computeAllGraphGradients;

computeAllGraphGradients = function(outputId) {
  const cache = ensureGraphStructureCache();
  const topo = cachedTopoForOutput(cache, outputId);

  // Stateful/special execution inside a loss expression is unusual and can have
  // side effects. Preserve the old evaluator for those cases rather than changing
  // semantics just for optimization.
  for (const id of topo) {
    const node = graph.nodes.get(id);
    const special = node ? getBlockDef(node.type).special : null;
    if (special === 'repeat' || special === 'setVariable' || special === 'derivative') {
      return computeAllGraphGradientsBeforeStructureCache(outputId);
    }
  }

  // Linear forward pass over the cached topo order. Every input is an O(1) array
  // lookup instead of graph.connections.find(...), and no recursive re-traversal is
  // needed because all parent values already exist earlier in topo.
  const values = new Map();
  for (const id of topo) {
    const node = graph.nodes.get(id);
    if (!node) throw new Error('존재하지 않는 블록입니다.');
    const def = getBlockDef(node.type);
    const inputs = [];

    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const connection = cachedGraphInput(cache, id, inputIndex);
      if (!connection) throw new Error(`입력 '${def.inputs[inputIndex]}'이 연결되지 않았습니다.`);
      if (!values.has(connection.from)) throw new Error('그래프 계산 순서가 올바르지 않습니다.');
      inputs.push(values.get(connection.from));
    }

    values.set(id, def.compute(node, inputs));
  }

  const output = values.get(outputId);
  if (typeof output !== 'number') throw new Error('미분 블록의 식 출력은 숫자 하나여야 합니다.');

  const adjoints = new Map([[outputId, 1]]);
  for (let k = topo.length - 1; k >= 0; k--) {
    const id = topo[k];
    const upstream = adjoints.get(id);
    if (upstream == null) continue;

    const node = graph.nodes.get(id);
    const def = getBlockDef(node.type);
    if (!def.inputs.length) continue;

    const inputIds = [];
    const inputs = [];
    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const connection = cachedGraphInput(cache, id, inputIndex);
      if (!connection) throw new Error(`미분 중 입력 '${def.inputs[inputIndex]}'이 비어 있습니다.`);
      inputIds.push(connection.from);
      inputs.push(values.get(connection.from));
    }

    const grads = node.type.startsWith('custom:')
      ? userBlockVJP(USER_BLOCKS.get(node.type.slice(7)), inputs, upstream)
      : primitiveVJP(node.type, inputs, values.get(id), upstream);
    grads.forEach((gradient, inputIndex) => accumulateGrad(adjoints, inputIds[inputIndex], gradient));
  }

  const gradients = new Map();
  for (const id of topo) {
    const node = graph.nodes.get(id);
    if (node?.type !== 'variable') continue;
    const name = String(node.params.name || 'x');
    const gradient = adjoints.get(id) ?? zerosLike(values.get(id));
    const previous = gradients.get(name);
    gradients.set(name, previous == null ? gradient : addValues(previous, gradient));
  }

  return gradients;
};
