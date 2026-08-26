// Cache the immutable graph structure during long training runs.
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

// A structure version string, used by the compiled plans to notice that the user
// edited the graph and their cached plans are stale.
function graphStructureVersion(cache) {
  return `${cache.nodeCount}:${cache.connectionCount}:${cache.hashA}:${cache.hashB}`;
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

function collectTopo(outputId) {
  return cachedTopoForOutput(ensureGraphStructureCache(), outputId);
}

// Every node id reachable backwards from `outputId`, including itself.
function collectAncestorNodeIds(outputId) {
  const cache = ensureGraphStructureCache();
  const visited = new Set();
  const visit = id => {
    if (visited.has(id)) return;
    visited.add(id);
    const node = graph.nodes.get(id);
    if (!node) return;
    const def = getBlockDef(node.type);
    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const connection = cachedGraphInput(cache, id, inputIndex);
      if (connection) visit(connection.from);
    }
  };
  visit(outputId);
  return visited;
}
