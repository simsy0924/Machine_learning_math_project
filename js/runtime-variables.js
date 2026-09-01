// 변수 blocks with state, and the loop indices a 반복 block exposes.
//
// A 변수 block is stateful: its value lives in RUNTIME_VARIABLES and 값 바꾸기
// replaces it. A 반복 index is NOT stored there — it is execution-local, so it
// lives on a stack and an inner 반복 can use `i` while an outer one still
// exposes `epoch`.

// The signature string decides whether a stored runtime value still belongs to
// the 변수 block that named it, so its exact format is part of the .mmlab file
// format: a saved project stores the signature next to each trained weight.
// Building it costs a JSON.stringify, and the manual-backprop hot path asks for
// it on every gradient read and write, so remember the last answer per node and
// rebuild only when one of the eight params it depends on actually changed.
const VARIABLE_SIGNATURE_CACHE = new WeakMap();

function variableSignature(node) {
  const params = node.params;
  const cached = VARIABLE_SIGNATURE_CACHE.get(node);
  if (
    cached &&
    cached.mode === params.mode &&
    cached.value === params.value &&
    cached.length === params.length &&
    cached.rows === params.rows &&
    cached.cols === params.cols &&
    cached.init === params.init &&
    cached.seed === params.seed &&
    cached.scale === params.scale
  ) {
    return cached.signature;
  }

  const signature = JSON.stringify({
    mode: params.mode,
    value: params.value,
    length: params.length,
    rows: params.rows,
    cols: params.cols,
    init: params.init,
    seed: params.seed,
    scale: params.scale
  });

  VARIABLE_SIGNATURE_CACHE.set(node, {
    mode: params.mode,
    value: params.value,
    length: params.length,
    rows: params.rows,
    cols: params.cols,
    init: params.init,
    seed: params.seed,
    scale: params.scale,
    signature
  });
  return signature;
}

function initialVariableValue(node) {
  const mode = node.params.mode || 'scalar';
  if (mode === 'scalar') return Number(node.params.value);

  if (mode === 'matrix') {
    const rows = Math.max(1, Math.floor(Number(node.params.rows) || 1));
    const cols = Math.max(1, Math.floor(Number(node.params.cols) || 1));
    if (node.params.init === 'random') return randomArray([rows, cols], node.params.seed, node.params.scale ?? 0.05);
    const data = new Float32Array(rows * cols);
    data.fill(Number(node.params.value));
    return arrayValue(data, [rows, cols]);
  }

  const length = Math.max(1, Math.floor(Number(node.params.length) || 1));
  if (node.params.init === 'random') return randomArray([length], node.params.seed, node.params.scale ?? 0.05);
  const data = new Float32Array(length);
  data.fill(Number(node.params.value));
  return arrayValue(data, [length]);
}

function findVariableNode(name) {
  for (const node of graph.nodes.values()) {
    if (node.type === 'variable' && String(node.params.name || 'x') === String(name)) return node;
  }
  return null;
}

// Runtime variable values are shared, not cloned. Values are immutable by rule
// (see README), so reading a weight matrix hands back the stored array instead
// of copying it on every access inside the training loop.
function readRuntimeVariable(node) {
  const name = String(node.params.name || 'x');
  const signature = variableSignature(node);
  const stored = RUNTIME_VARIABLES.get(name);
  if (!stored || stored.signature !== signature) {
    const value = initialVariableValue(node);
    RUNTIME_VARIABLES.set(name, { value, signature });
    return value;
  }
  return stored.value;
}

function writeRuntimeVariable(name, value, immediate = false) {
  const variableNode = findVariableNode(name);
  if (!variableNode) throw new Error(`'${name}'이라는 변수 블록을 찾지 못했습니다.`);
  const entry = { value, signature: variableSignature(variableNode) };
  if (pendingVariableUpdates && !immediate) pendingVariableUpdates.set(String(name), entry);
  else RUNTIME_VARIABLES.set(String(name), entry);
  return value;
}

function commitPendingVariableUpdates() {
  if (!pendingVariableUpdates) return;
  for (const [name, entry] of pendingVariableUpdates) RUNTIME_VARIABLES.set(name, entry);
}

// ---------- loop indices ----------

const LOOP_CONTEXT_STACK = [];

function activeLoopValue(name) {
  const key = String(name);
  for (let i = LOOP_CONTEXT_STACK.length - 1; i >= 0; i--) {
    const frame = LOOP_CONTEXT_STACK[i];
    if (frame.has(key)) return { found: true, value: frame.get(key) };
  }
  return { found: false, value: undefined };
}

// A 변수 block resolves to the active loop index when one shadows its name,
// otherwise to its stored runtime value.
function variableBlockValue(node) {
  const loopValue = activeLoopValue(node.params.name || 'x');
  if (loopValue.found) return loopValue.value;
  return readRuntimeVariable(node);
}
