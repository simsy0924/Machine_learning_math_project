const workspace = document.getElementById('workspace');
const nodesLayer = document.getElementById('nodes');
const wiresSvg = document.getElementById('wires');
const emptyState = document.getElementById('emptyState');
const workspaceStatus = document.getElementById('workspaceStatus');
const connectionHint = document.getElementById('connectionHint');
const evaluateBtn = document.getElementById('evaluateBtn');
const resetWorkspaceBtn = document.getElementById('resetWorkspaceBtn');

const inspectorEmpty = document.getElementById('inspectorEmpty');
const inspectorContent = document.getElementById('inspectorContent');
const inspectorTitle = document.getElementById('inspectorTitle');
const inspectorDescription = document.getElementById('inspectorDescription');
const inspectorFormula = document.getElementById('inspectorFormula');
const inspectorValue = document.getElementById('inspectorValue');
const inspectorControls = document.getElementById('inspectorControls');
const deleteNodeBtn = document.getElementById('deleteNodeBtn');

const groupSelectBtn = document.getElementById('groupSelectBtn');
const createGroupBtn = document.getElementById('createGroupBtn');
const cancelGroupBtn = document.getElementById('cancelGroupBtn');
const myBlocksPalette = document.getElementById('myBlocksPalette');
const myBlocksEmpty = document.getElementById('myBlocksEmpty');

const drawCanvas = document.getElementById('drawCanvas');
const drawCtx = drawCanvas.getContext('2d', { willReadFrequently: true });
const previewCanvas = document.getElementById('previewCanvas');
const previewCtx = previewCanvas.getContext('2d');
const clearDrawBtn = document.getElementById('clearDrawBtn');

const DATA_CLASS_OPTIONS = [
  ['cat', '고양이'], ['fish', '물고기'], ['house', '집'], ['tree', '나무'],
  ['car', '자동차'], ['apple', '사과'], ['clock', '시계'], ['star', '별'],
  ['umbrella', '우산'], ['airplane', '비행기'], ['face', '얼굴'], ['flower', '꽃'],
  ['cup', '컵'], ['bicycle', '자전거'], ['guitar', '기타']
];

let nextNodeId = 1;
let selectedNodeId = null;
let pendingOutput = null;
let drawing = false;
let lastPoint = { x: 0, y: 0 };
let groupSelectionMode = false;
const groupSelectedIds = new Set();

const graph = { nodes: new Map(), connections: [] };
const USER_BLOCKS = new Map();
const USER_BLOCK_STORAGE_KEY = 'machine-learning-math-project.user-blocks.v1';

function arrayValue(data, shape = [data.length]) {
  return { kind: 'array', data: data instanceof Float32Array ? data : new Float32Array(data), shape: [...shape] };
}
// Internal constructor for the math kernels below. The caller always hands over a
// freshly allocated buffer, and shape arrays are never mutated anywhere in the
// runtime, so both can be adopted directly instead of being copied on every
// single operation.
function fastArrayValue(data, shape) {
  return { kind: 'array', data, shape };
}
function isArrayValue(value) { return Boolean(value && value.kind === 'array'); }
function asArrayValue(value) {
  if (!isArrayValue(value)) throw new Error('배열 입력이 필요합니다.');
  return value;
}
function copyValue(value) {
  if (typeof value === 'number') return value;
  if (isArrayValue(value)) return arrayValue(new Float32Array(value.data), value.shape);
  return value;
}
function elementwiseUnary(value, fn) {
  if (typeof value === 'number') return fn(value);
  const arr = asArrayValue(value);
  const source = arr.data;
  const data = new Float32Array(source.length);
  for (let i = 0; i < data.length; i++) data[i] = fn(source[i], i);
  return fastArrayValue(data, arr.shape);
}
function elementwiseBinary(a, b, fn) {
  if (typeof a === 'number' && typeof b === 'number') return fn(a, b);
  // Each mixed case gets its own loop. Wrapping fn in another closure here used
  // to add a second layer of indirect calls to every element.
  if (typeof a === 'number') {
    const bb = asArrayValue(b), bd = bb.data;
    const data = new Float32Array(bd.length);
    for (let i = 0; i < data.length; i++) data[i] = fn(a, bd[i]);
    return fastArrayValue(data, bb.shape);
  }
  const aa = asArrayValue(a), ad = aa.data;
  if (typeof b === 'number') {
    const data = new Float32Array(ad.length);
    for (let i = 0; i < data.length; i++) data[i] = fn(ad[i], b);
    return fastArrayValue(data, aa.shape);
  }
  const bd = asArrayValue(b).data;
  if (ad.length !== bd.length) throw new Error('배열의 원소 수가 서로 다릅니다.');
  const data = new Float32Array(ad.length);
  for (let i = 0; i < data.length; i++) data[i] = fn(ad[i], bd[i]);
  return fastArrayValue(data, aa.shape);
}

// ---------- 원소별 연산 커널 ----------
// These are the innermost loops of both the forward pass and backpropagation. A
// single SGD step on a 15x784 weight matrix runs tens of thousands of elements
// through them, so they are written as plain typed loops instead of going
// through elementwiseBinary/elementwiseUnary with a per-element callback.
// elementwiseBinary and elementwiseUnary remain for user-defined blocks and any
// operation without a dedicated kernel.

function addValues(a, b) {
  if (typeof a === 'number') {
    if (typeof b === 'number') return a + b;
    const bb = asArrayValue(b), bd = bb.data;
    const out = new Float32Array(bd.length);
    for (let i = 0; i < out.length; i++) out[i] = a + bd[i];
    return fastArrayValue(out, bb.shape);
  }
  const aa = asArrayValue(a), ad = aa.data;
  if (typeof b === 'number') {
    const out = new Float32Array(ad.length);
    for (let i = 0; i < out.length; i++) out[i] = ad[i] + b;
    return fastArrayValue(out, aa.shape);
  }
  const bd = asArrayValue(b).data;
  if (ad.length !== bd.length) throw new Error('배열의 원소 수가 서로 다릅니다.');
  const out = new Float32Array(ad.length);
  for (let i = 0; i < out.length; i++) out[i] = ad[i] + bd[i];
  return fastArrayValue(out, aa.shape);
}

function subtractValues(a, b) {
  if (typeof a === 'number') {
    if (typeof b === 'number') return a - b;
    const bb = asArrayValue(b), bd = bb.data;
    const out = new Float32Array(bd.length);
    for (let i = 0; i < out.length; i++) out[i] = a - bd[i];
    return fastArrayValue(out, bb.shape);
  }
  const aa = asArrayValue(a), ad = aa.data;
  if (typeof b === 'number') {
    const out = new Float32Array(ad.length);
    for (let i = 0; i < out.length; i++) out[i] = ad[i] - b;
    return fastArrayValue(out, aa.shape);
  }
  const bd = asArrayValue(b).data;
  if (ad.length !== bd.length) throw new Error('배열의 원소 수가 서로 다릅니다.');
  const out = new Float32Array(ad.length);
  for (let i = 0; i < out.length; i++) out[i] = ad[i] - bd[i];
  return fastArrayValue(out, aa.shape);
}

function multiplyValues(a, b) {
  if (typeof a === 'number') {
    if (typeof b === 'number') return a * b;
    const bb = asArrayValue(b), bd = bb.data;
    const out = new Float32Array(bd.length);
    for (let i = 0; i < out.length; i++) out[i] = a * bd[i];
    return fastArrayValue(out, bb.shape);
  }
  const aa = asArrayValue(a), ad = aa.data;
  if (typeof b === 'number') {
    const out = new Float32Array(ad.length);
    for (let i = 0; i < out.length; i++) out[i] = ad[i] * b;
    return fastArrayValue(out, aa.shape);
  }
  const bd = asArrayValue(b).data;
  if (ad.length !== bd.length) throw new Error('배열의 원소 수가 서로 다릅니다.');
  const out = new Float32Array(ad.length);
  for (let i = 0; i < out.length; i++) out[i] = ad[i] * bd[i];
  return fastArrayValue(out, aa.shape);
}

function divideValues(a, b) {
  if (typeof a === 'number') {
    if (typeof b === 'number') return a / b;
    const bb = asArrayValue(b), bd = bb.data;
    const out = new Float32Array(bd.length);
    for (let i = 0; i < out.length; i++) out[i] = a / bd[i];
    return fastArrayValue(out, bb.shape);
  }
  const aa = asArrayValue(a), ad = aa.data;
  if (typeof b === 'number') {
    const out = new Float32Array(ad.length);
    for (let i = 0; i < out.length; i++) out[i] = ad[i] / b;
    return fastArrayValue(out, aa.shape);
  }
  const bd = asArrayValue(b).data;
  if (ad.length !== bd.length) throw new Error('배열의 원소 수가 서로 다릅니다.');
  const out = new Float32Array(ad.length);
  for (let i = 0; i < out.length; i++) out[i] = ad[i] / bd[i];
  return fastArrayValue(out, aa.shape);
}

function maximumValues(a, b) {
  if (typeof a === 'number') {
    if (typeof b === 'number') return Math.max(a, b);
    const bb = asArrayValue(b), bd = bb.data;
    const out = new Float32Array(bd.length);
    for (let i = 0; i < out.length; i++) out[i] = Math.max(a, bd[i]);
    return fastArrayValue(out, bb.shape);
  }
  const aa = asArrayValue(a), ad = aa.data;
  if (typeof b === 'number') {
    const out = new Float32Array(ad.length);
    for (let i = 0; i < out.length; i++) out[i] = Math.max(ad[i], b);
    return fastArrayValue(out, aa.shape);
  }
  const bd = asArrayValue(b).data;
  if (ad.length !== bd.length) throw new Error('배열의 원소 수가 서로 다릅니다.');
  const out = new Float32Array(ad.length);
  for (let i = 0; i < out.length; i++) out[i] = Math.max(ad[i], bd[i]);
  return fastArrayValue(out, aa.shape);
}

function equalValues(a, b) {
  if (typeof a === 'number') {
    if (typeof b === 'number') return a === b ? 1 : 0;
    const bb = asArrayValue(b), bd = bb.data;
    const out = new Float32Array(bd.length);
    for (let i = 0; i < out.length; i++) out[i] = a === bd[i] ? 1 : 0;
    return fastArrayValue(out, bb.shape);
  }
  const aa = asArrayValue(a), ad = aa.data;
  if (typeof b === 'number') {
    const out = new Float32Array(ad.length);
    for (let i = 0; i < out.length; i++) out[i] = ad[i] === b ? 1 : 0;
    return fastArrayValue(out, aa.shape);
  }
  const bd = asArrayValue(b).data;
  if (ad.length !== bd.length) throw new Error('배열의 원소 수가 서로 다릅니다.');
  const out = new Float32Array(ad.length);
  for (let i = 0; i < out.length; i++) out[i] = ad[i] === bd[i] ? 1 : 0;
  return fastArrayValue(out, aa.shape);
}

function negateValue(v) {
  if (typeof v === 'number') return -v;
  const a = asArrayValue(v), d = a.data;
  const out = new Float32Array(d.length);
  for (let i = 0; i < out.length; i++) out[i] = -d[i];
  return fastArrayValue(out, a.shape);
}

function squareValues(v) {
  if (typeof v === 'number') return v * v;
  const a = asArrayValue(v), d = a.data;
  const out = new Float32Array(d.length);
  for (let i = 0; i < out.length; i++) out[i] = d[i] * d[i];
  return fastArrayValue(out, a.shape);
}

function absValues(v) {
  if (typeof v === 'number') return Math.abs(v);
  const a = asArrayValue(v), d = a.data;
  const out = new Float32Array(d.length);
  for (let i = 0; i < out.length; i++) out[i] = Math.abs(d[i]);
  return fastArrayValue(out, a.shape);
}

function expValues(v) {
  if (typeof v === 'number') return Math.exp(v);
  const a = asArrayValue(v), d = a.data;
  const out = new Float32Array(d.length);
  for (let i = 0; i < out.length; i++) out[i] = Math.exp(d[i]);
  return fastArrayValue(out, a.shape);
}

function logValues(v) {
  if (typeof v === 'number') return Math.log(v);
  const a = asArrayValue(v), d = a.data;
  const out = new Float32Array(d.length);
  for (let i = 0; i < out.length; i++) out[i] = Math.log(d[i]);
  return fastArrayValue(out, a.shape);
}
function sumArray(arr) { let s = 0; for (const v of arr.data) s += v; return s; }
function zerosLike(v) {
  if (typeof v === 'number') return 0;
  if (isArrayValue(v)) return fastArrayValue(new Float32Array(v.data.length), v.shape);
  return null;
}
function fillLike(v, scalar) {
  if (typeof v === 'number') return scalar;
  const a = asArrayValue(v);
  const data = new Float32Array(a.data.length);
  data.fill(scalar);
  return fastArrayValue(data, a.shape);
}
function unbroadcast(grad, original) {
  if (typeof original === 'number' && isArrayValue(grad)) return sumArray(grad);
  if (isArrayValue(original) && typeof grad === 'number') return fillLike(original, grad);
  // Same element count, different shape: reinterpret instead of copying the
  // whole buffer. This runs on every backward pass through 행렬×벡터 and 더하기.
  if (isArrayValue(original) && isArrayValue(grad)) return fastArrayValue(grad.data, original.shape);
  return grad;
}
function accumulateGrad(map, key, grad) {
  if (grad == null) return;
  map.set(key, map.has(key) ? addValues(map.get(key), grad) : copyValue(grad));
}
function dotValues(a, b) {
  const aa = asArrayValue(a), bb = asArrayValue(b);
  if (aa.data.length !== bb.data.length) throw new Error('내적할 두 벡터의 길이가 다릅니다.');
  let s = 0;
  for (let i = 0; i < aa.data.length; i++) s += aa.data[i] * bb.data[i];
  return s;
}
function deterministicRandomVector(length, seed) {
  let state = (Number(seed) || 1) >>> 0;
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    state = (1664525 * state + 1013904223) >>> 0;
    out[i] = (state / 4294967296) * 2 - 1;
  }
  return arrayValue(out, [length]);
}
function classOptions() { return DATA_CLASS_OPTIONS.map(([value, label]) => ({ value, label })); }
