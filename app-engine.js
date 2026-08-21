function evaluateGraph() {
  const memo = new Map(), visiting = new Set(); let successCount = 0;
  for (const node of graph.nodes.values()) { node.lastValue = undefined; node.lastError = null; }
  for (const node of graph.nodes.values()) {
    try { node.lastValue = evaluateNode(node.id, memo, visiting); successCount++; } catch (error) { node.lastError = error.message; }
    updateNodePreview(node);
  }
  renderInspector(); workspaceStatus.textContent = `${successCount}/${graph.nodes.size} 계산`;
}

function evaluateNode(nodeId, memo = new Map(), visiting = new Set()) {
  if (memo.has(nodeId)) return memo.get(nodeId);
  if (visiting.has(nodeId)) throw new Error('순환 연결은 계산할 수 없습니다.');
  const node = graph.nodes.get(nodeId); if (!node) throw new Error('존재하지 않는 블록입니다.');
  const def = getBlockDef(node.type); visiting.add(nodeId);
  if (def.special === 'derivative') {
    const c = graphInputConnection(nodeId, 0); if (!c) throw new Error("입력 '식'이 연결되지 않았습니다.");
    const value = differentiateGraph(c.from, String(node.params.variable || 'x'));
    memo.set(nodeId, value); visiting.delete(nodeId); return value;
  }
  const inputs = def.inputs.map((label, inputIndex) => { const c = graphInputConnection(nodeId, inputIndex); if (!c) throw new Error(`입력 '${label}'이 연결되지 않았습니다.`); return evaluateNode(c.from, memo, visiting); });
  const value = def.compute(node, inputs); memo.set(nodeId, value); visiting.delete(nodeId); return value;
}

function collectTopo(outputId) {
  const seen = new Set(), order = [];
  const visit = id => { if (seen.has(id)) return; seen.add(id); const node = graph.nodes.get(id); if (!node) return; const def = getBlockDef(node.type); for (let i = 0; i < def.inputs.length; i++) { const c = graph.connections.find(x => x.to === id && x.inputIndex === i); if (c) visit(c.from); } order.push(id); };
  visit(outputId); return order;
}

// Reverse-mode autodiff naturally produces gradients for every variable in one
// backward pass. Derivative blocks that point at the same scalar loss therefore
// share one pass while the training state (weights + loop indices) is unchanged.
//
// Gradients are handed out without copying. Values in this runtime are immutable
// by rule (see README), so the derivative blocks sharing a pass can share the
// same arrays, and a shared 15x784 weight gradient is not cloned per block.
const AUTODIFF_ENTRY_IDS = new WeakMap();
let nextAutodiffEntryId = 1;
let sharedAutodiffContextKey = null;
let sharedAutodiffByOutput = new Map();

function autodiffEntryId(entry) {
  if (!entry || (typeof entry !== 'object' && typeof entry !== 'function')) return String(entry);
  let id = AUTODIFF_ENTRY_IDS.get(entry);
  if (id == null) {
    id = nextAutodiffEntryId++;
    AUTODIFF_ENTRY_IDS.set(entry, id);
  }
  return id;
}

function currentTrainingAutodiffContextKey() {
  // Sharing is intentionally limited to active repeat execution. Outside a
  // repeat, inputs can change from UI edits without an execution-state version.
  if (typeof LOOP_CONTEXT_STACK === 'undefined' || !LOOP_CONTEXT_STACK.length) return null;

  const loopParts = [];
  for (const frame of LOOP_CONTEXT_STACK) {
    const entries = Array.from(frame.entries())
      .map(([name, value]) => `${String(name)}=${Number(value)}`)
      .sort();
    loopParts.push(entries.join(','));
  }

  const runtimeParts = [];
  if (typeof RUNTIME_VARIABLES !== 'undefined') {
    for (const [name, entry] of RUNTIME_VARIABLES.entries()) {
      runtimeParts.push(`${String(name)}#${autodiffEntryId(entry)}`);
    }
    runtimeParts.sort();
  }

  return `${loopParts.join('/')};${runtimeParts.join('|')}`;
}

function computeAllGraphGradients(outputId) {
  const topo = collectTopo(outputId), values = new Map(), memo = new Map();
  for (const id of topo) values.set(id, evaluateNode(id, memo, new Set()));
  const output = values.get(outputId); if (typeof output !== 'number') throw new Error('미분 블록의 식 출력은 숫자 하나여야 합니다.');

  const adjoints = new Map([[outputId, 1]]);
  for (let k = topo.length - 1; k >= 0; k--) {
    const id = topo[k], upstream = adjoints.get(id); if (upstream == null) continue;
    const node = graph.nodes.get(id), def = getBlockDef(node.type); if (!def.inputs.length || def.special === 'derivative') continue;
    const inputIds = [], inputs = [];
    for (let i = 0; i < def.inputs.length; i++) { const c = graphInputConnection(id, i); if (!c) throw new Error(`미분 중 입력 '${def.inputs[i]}'이 비어 있습니다.`); inputIds.push(c.from); inputs.push(values.get(c.from)); }
    const grads = node.type.startsWith('custom:') ? userBlockVJP(USER_BLOCKS.get(node.type.slice(7)), inputs, upstream) : primitiveVJP(node.type, inputs, values.get(id), upstream);
    grads.forEach((g, i) => accumulateGrad(adjoints, inputIds[i], g));
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
}

function differentiateGraph(outputId, variableName) {
  const requestedName = String(variableName);
  const contextBefore = currentTrainingAutodiffContextKey();

  if (contextBefore != null) {
    if (sharedAutodiffContextKey !== contextBefore) {
      sharedAutodiffContextKey = contextBefore;
      sharedAutodiffByOutput = new Map();
    }
    const cached = sharedAutodiffByOutput.get(outputId);
    if (cached) {
      if (!cached.has(requestedName)) throw new Error(`'${requestedName}'이라는 변수를 식에서 찾지 못했습니다.`);
      return cached.get(requestedName);
    }
  }

  const gradients = computeAllGraphGradients(outputId);
  if (!gradients.has(requestedName)) throw new Error(`'${requestedName}'이라는 변수를 식에서 찾지 못했습니다.`);

  if (contextBefore != null) {
    // Some variables are lazily initialized during the first forward pass, so
    // cache under the post-forward state identity rather than the pre-pass key.
    const contextAfter = currentTrainingAutodiffContextKey();
    sharedAutodiffContextKey = contextAfter;
    sharedAutodiffByOutput = new Map([[outputId, gradients]]);
  }

  return gradients.get(requestedName);
}

function primitiveVJP(type, inputs, output, upstream) {
  const [a, b] = inputs;
  switch (type) {
    case 'add': return [unbroadcast(upstream, a), unbroadcast(upstream, b)];
    case 'subtract': return [unbroadcast(upstream, a), unbroadcast(negateValue(upstream), b)];
    case 'multiply': return [unbroadcast(multiplyValues(upstream, b), a), unbroadcast(multiplyValues(upstream, a), b)];
    case 'divide': return [unbroadcast(divideValues(upstream, b), a), unbroadcast(negateValue(divideValues(multiplyValues(upstream, a), multiplyValues(b, b))), b)];
    case 'square': return [multiplyValues(upstream, multiplyValues(a, 2))];
    case 'abs': return [multiplyValues(upstream, elementwiseUnary(a, x => x > 0 ? 1 : x < 0 ? -1 : 0))];
    case 'maximum': {
      const fa = elementwiseBinary(a, b, (x, y) => x > y ? 1 : x < y ? 0 : 0.5); const fb = elementwiseBinary(a, b, (x, y) => x < y ? 1 : x > y ? 0 : 0.5);
      return [unbroadcast(multiplyValues(upstream, fa), a), unbroadcast(multiplyValues(upstream, fb), b)];
    }
    case 'exp': return [multiplyValues(upstream, output)];
    case 'log': return [divideValues(upstream, a)];
    case 'sum': return [fillLike(a, typeof upstream === 'number' ? upstream : sumArray(asArrayValue(upstream)))];
    case 'mean': { const n = typeof a === 'number' ? 1 : asArrayValue(a).data.length; return [fillLike(a, (typeof upstream === 'number' ? upstream : sumArray(asArrayValue(upstream))) / n)]; }
    case 'dot': return [multiplyValues(b, upstream), multiplyValues(a, upstream)];
    case 'flatten': return [isArrayValue(a) ? arrayValue(new Float32Array(asArrayValue(upstream).data), a.shape) : upstream];
    case 'display': return [upstream];
    default: return inputs.map(() => null);
  }
}

function evaluateUserDefinition(definition, externalValues) {
  const values = new Map(), visiting = new Set();
  const externalBySlot = new Map(definition.externalInputs.map((slot, i) => [`${slot.nodeId}:${slot.inputIndex ?? 'source'}`, externalValues[i]]));
  const nodeMap = new Map(definition.nodes.map(n => [n.id, n]));
  const evalInternal = id => {
    if (values.has(id)) return values.get(id); if (visiting.has(id)) throw new Error('사용자 블록 내부에 순환 연결이 있습니다.'); visiting.add(id);
    const node = nodeMap.get(id); if (!node) throw new Error('사용자 블록 내부 노드를 찾을 수 없습니다.');
    const sourceKey = `${id}:source`; if (externalBySlot.has(sourceKey)) { const v = externalBySlot.get(sourceKey); values.set(id, v); visiting.delete(id); return v; }
    const blockDef = getBlockDef(node.type); if (blockDef.special === 'derivative') throw new Error('현재 미분 블록 자체는 사용자 블록 내부에 넣을 수 없습니다.');
    const ins = blockDef.inputs.map((label, inputIndex) => { const c = definition.connections.find(x => x.to === id && x.inputIndex === inputIndex); if (c) return evalInternal(c.from); const key = `${id}:${inputIndex}`; if (!externalBySlot.has(key)) throw new Error(`사용자 블록 입력 '${label}'이 비어 있습니다.`); return externalBySlot.get(key); });
    const v = blockDef.compute(node, ins); values.set(id, v); visiting.delete(id); return v;
  };
  return evalInternal(definition.outputNodeId);
}

function userBlockVJP(definition, externalValues, upstream) {
  const nodeMap = new Map(definition.nodes.map(n => [n.id, n])), values = new Map(), topo = [], seen = new Set();
  const externalIndexBySlot = new Map(definition.externalInputs.map((slot, i) => [`${slot.nodeId}:${slot.inputIndex ?? 'source'}`, i]));
  const visit = id => { if (seen.has(id)) return; seen.add(id); const node = nodeMap.get(id), blockDef = getBlockDef(node.type); if (!externalIndexBySlot.has(`${id}:source`)) { for (let i = 0; i < blockDef.inputs.length; i++) { const c = definition.connections.find(x => x.to === id && x.inputIndex === i); if (c) visit(c.from); } } topo.push(id); };
  visit(definition.outputNodeId);
  const evalInternal = id => {
    if (values.has(id)) return values.get(id); const node = nodeMap.get(id); const sourceSlot = externalIndexBySlot.get(`${id}:source`); if (sourceSlot != null) { const v = externalValues[sourceSlot]; values.set(id, v); return v; }
    const blockDef = getBlockDef(node.type); const ins = blockDef.inputs.map((label, i) => { const c = definition.connections.find(x => x.to === id && x.inputIndex === i); if (c) return evalInternal(c.from); const slot = externalIndexBySlot.get(`${id}:${i}`); if (slot == null) throw new Error('사용자 블록의 외부 입력을 찾지 못했습니다.'); return externalValues[slot]; }); const v = blockDef.compute(node, ins); values.set(id, v); return v;
  };
  evalInternal(definition.outputNodeId);
  const adj = new Map([[definition.outputNodeId, upstream]]), externalGrads = externalValues.map(v => zerosLike(v));
  for (let k = topo.length - 1; k >= 0; k--) {
    const id = topo[k], up = adj.get(id); if (up == null) continue; const node = nodeMap.get(id); const blockDef = getBlockDef(node.type); const sourceSlot = externalIndexBySlot.get(`${id}:source`); if (sourceSlot != null) { externalGrads[sourceSlot] = externalGrads[sourceSlot] == null ? up : addValues(externalGrads[sourceSlot], up); continue; }
    const ins = [], refs = [];
    for (let i = 0; i < blockDef.inputs.length; i++) { const c = definition.connections.find(x => x.to === id && x.inputIndex === i); if (c) { ins.push(values.get(c.from)); refs.push({ nodeId: c.from }); } else { const slot = externalIndexBySlot.get(`${id}:${i}`); ins.push(externalValues[slot]); refs.push({ external: slot }); } }
    const grads = node.type.startsWith('custom:') ? userBlockVJP(USER_BLOCKS.get(node.type.slice(7)), ins, up) : primitiveVJP(node.type, ins, values.get(id), up);
    grads.forEach((g, i) => { if (g == null) return; const ref = refs[i]; if (ref.external != null) externalGrads[ref.external] = externalGrads[ref.external] == null ? g : addValues(externalGrads[ref.external], g); else accumulateGrad(adj, ref.nodeId, g); });
  }
  return externalGrads;
}
