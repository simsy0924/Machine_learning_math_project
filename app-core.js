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
  const data = new Float32Array(arr.data.length);
  for (let i = 0; i < data.length; i++) data[i] = fn(arr.data[i], i);
  return arrayValue(data, arr.shape);
}
function elementwiseBinary(a, b, fn) {
  if (typeof a === 'number' && typeof b === 'number') return fn(a, b);
  if (typeof a === 'number' && isArrayValue(b)) return elementwiseUnary(b, x => fn(a, x));
  if (isArrayValue(a) && typeof b === 'number') return elementwiseUnary(a, x => fn(x, b));
  const aa = asArrayValue(a);
  const bb = asArrayValue(b);
  if (aa.data.length !== bb.data.length) throw new Error('배열의 원소 수가 서로 다릅니다.');
  const data = new Float32Array(aa.data.length);
  for (let i = 0; i < data.length; i++) data[i] = fn(aa.data[i], bb.data[i]);
  return arrayValue(data, aa.shape);
}
function addValues(a, b) { return elementwiseBinary(a, b, (x, y) => x + y); }
function subtractValues(a, b) { return elementwiseBinary(a, b, (x, y) => x - y); }
function multiplyValues(a, b) { return elementwiseBinary(a, b, (x, y) => x * y); }
function divideValues(a, b) { return elementwiseBinary(a, b, (x, y) => x / y); }
function negateValue(v) { return elementwiseUnary(v, x => -x); }
function sumArray(arr) { let s = 0; for (const v of arr.data) s += v; return s; }
function zerosLike(v) {
  if (typeof v === 'number') return 0;
  if (isArrayValue(v)) { const a = asArrayValue(v); return arrayValue(new Float32Array(a.data.length), a.shape); }
  return null;
}
function fillLike(v, scalar) {
  if (typeof v === 'number') return scalar;
  const a = asArrayValue(v);
  const data = new Float32Array(a.data.length);
  data.fill(scalar);
  return arrayValue(data, a.shape);
}
function unbroadcast(grad, original) {
  if (typeof original === 'number' && isArrayValue(grad)) return sumArray(grad);
  if (isArrayValue(original) && typeof grad === 'number') return fillLike(original, grad);
  if (isArrayValue(original) && isArrayValue(grad)) return arrayValue(new Float32Array(grad.data), original.shape);
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
