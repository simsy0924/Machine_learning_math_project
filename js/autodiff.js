// Reverse-mode automatic differentiation over the block graph.
//
// primitiveVJP is a plain lookup: each block definition carries its own `vjp`
// next to its forward pass (see js/blocks.js). Nothing here has to be edited to
// add a differentiable block.
//
// computeAllGraphGradients picks an execution strategy. Compiled strategies are
// registered from their own files (js/autodiff-compiled.js,
// js/autodiff-spatial-fusion.js) and are tried in priority order; the
// interpreted implementations below are the always-available fallback.

function primitiveVJP(type, inputs, output, upstream) {
  const vjp = BLOCKS[type]?.vjp;
  if (!vjp) return inputs.map(() => null);

  const hook = PROFILER_HOOKS.vjp;
  if (!hook || !hook.active()) return vjp(inputs, output, upstream);

  const started = performance.now();
  try {
    return vjp(inputs, output, upstream);
  } finally {
    hook.record(type, performance.now() - started);
  }
}

// Gradient of every 변수 in `topo`, given the adjoints of its nodes. Several
// 변수 blocks may share one name, so their gradients are summed.
function collectVariableGradients(topo, values, adjoints) {
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

// Interpreted pass for loss expressions that contain stateful or special blocks.
// Those can have side effects and need the full evaluator, so the forward pass
// goes through evaluateNode instead of a straight-line walk.
function statefulGraphGradients(outputId) {
  const topo = collectTopo(outputId);
  const values = new Map();
  const memo = new Map();
  for (const id of topo) values.set(id, evaluateNode(id, memo, new Set()));

  const output = values.get(outputId);
  if (typeof output !== 'number') throw new Error('미분 블록의 식 출력은 숫자 하나여야 합니다.');

  const adjoints = new Map([[outputId, 1]]);
  for (let k = topo.length - 1; k >= 0; k--) {
    const id = topo[k];
    const upstream = adjoints.get(id);
    if (upstream == null) continue;

    const node = graph.nodes.get(id);
    const def = getBlockDef(node.type);
    if (!def.inputs.length || def.special === 'derivative') continue;

    const inputIds = [];
    const inputs = [];
    for (let i = 0; i < def.inputs.length; i++) {
      const connection = graphInputConnection(id, i);
      if (!connection) throw new Error(`미분 중 입력 '${def.inputs[i]}'이 비어 있습니다.`);
      inputIds.push(connection.from);
      inputs.push(values.get(connection.from));
    }

    const grads = node.type.startsWith('custom:')
      ? userBlockVJP(USER_BLOCKS.get(node.type.slice(7)), inputs, upstream)
      : primitiveVJP(node.type, inputs, values.get(id), upstream);
    grads.forEach((g, i) => accumulateGrad(adjoints, inputIds[i], g));
  }

  return collectVariableGradients(topo, values, adjoints);
}

// Interpreted pass for an ordinary pure loss expression. Every input is an O(1)
// array lookup instead of graph.connections.find(...), and no recursive
// re-traversal is needed because parent values already exist earlier in topo.
function interpretedGraphGradients(outputId, structureCache) {
  const topo = cachedTopoForOutput(structureCache, outputId);

  for (const id of topo) {
    const node = graph.nodes.get(id);
    const special = node ? getBlockDef(node.type).special : null;
    if (special === 'repeat' || special === 'setVariable' || special === 'derivative') {
      return statefulGraphGradients(outputId);
    }
  }

  const values = new Map();
  for (const id of topo) {
    const node = graph.nodes.get(id);
    if (!node) throw new Error('존재하지 않는 블록입니다.');
    const def = getBlockDef(node.type);
    const inputs = [];

    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const connection = cachedGraphInput(structureCache, id, inputIndex);
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
      const connection = cachedGraphInput(structureCache, id, inputIndex);
      if (!connection) throw new Error(`미분 중 입력 '${def.inputs[inputIndex]}'이 비어 있습니다.`);
      inputIds.push(connection.from);
      inputs.push(values.get(connection.from));
    }

    const grads = node.type.startsWith('custom:')
      ? userBlockVJP(USER_BLOCKS.get(node.type.slice(7)), inputs, upstream)
      : primitiveVJP(node.type, inputs, values.get(id), upstream);
    grads.forEach((gradient, inputIndex) => accumulateGrad(adjoints, inputIds[inputIndex], gradient));
  }

  return collectVariableGradients(topo, values, adjoints);
}

function runGradientStrategies(outputId) {
  const structureCache = ensureGraphStructureCache();
  for (const strategy of GRADIENT_STRATEGIES) {
    const plan = strategy.plan(outputId, structureCache);
    if (plan) return strategy.execute(plan);
  }
  return interpretedGraphGradients(outputId, structureCache);
}

function computeAllGraphGradients(outputId) {
  const hook = PROFILER_HOOKS.gradientPass;
  if (!hook) return withUserBlockGradientCapture(() => runGradientStrategies(outputId));

  const token = hook.begin();
  try {
    return withUserBlockGradientCapture(() => runGradientStrategies(outputId));
  } finally {
    hook.end(token);
  }
}

// ---------- shared gradient passes ----------
//
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
  if (!LOOP_CONTEXT_STACK.length) return null;

  const loopParts = [];
  for (const frame of LOOP_CONTEXT_STACK) {
    const entries = Array.from(frame.entries())
      .map(([name, value]) => `${String(name)}=${Number(value)}`)
      .sort();
    loopParts.push(entries.join(','));
  }

  const runtimeParts = [];
  for (const [name, entry] of RUNTIME_VARIABLES.entries()) {
    runtimeParts.push(`${String(name)}#${autodiffEntryId(entry)}`);
  }
  runtimeParts.sort();

  return `${loopParts.join('/')};${runtimeParts.join('|')}`;
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
