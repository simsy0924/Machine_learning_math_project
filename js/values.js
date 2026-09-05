// The value model and its numeric kernels.
//
// A value is either a plain JavaScript number or an array value:
//   { kind: 'array', data: Float32Array, shape: number[] }
// Array values are IMMUTABLE by rule. Nothing in the runtime writes into a
// buffer it did not just allocate, which is what lets reshape share a buffer and
// lets a gradient be handed to several consumers without copying.

function arrayValue(data, shape = [data.length]) {
  return { kind: 'array', data: data instanceof Float32Array ? data : new Float32Array(data), shape: [...shape] };
}

// Internal constructor for the kernels below. The caller always hands over a
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

// ---------- 결과 버퍼 아레나 ----------
// A single SGD step on a 64×784 weight allocates the gradient, the scaled
// gradient and the updated weight — about 600 KB — and then throws nearly all
// of it away. Measured on that graph, allocating a fresh Float32Array and
// filling it costs about four times as much as writing into a buffer that
// already exists, because a new buffer is zero-filled by the VM and its pages
// are cold. So the compiled training and evaluation loops hand their kernels a
// pool of buffers to reuse.
//
// The correctness rule is that a buffer is recycled only once the iteration
// that produced it has finished AND its value is not reachable from runtime
// variable state. endResultArenaIteration decides that by inspecting the actual
// values at the iteration boundary rather than by static analysis, so blocks
// that pass an input buffer straight through — 펼치기, 둘 다 계산, 값 보기,
// 사용자 블록 — need no special handling.
//
// Only the kernels in this file draw from the arena. Anything whose result
// outlives one iteration (block caches, 변수 초기값, copyValue, dataset images)
// must keep allocating its own buffer.
let RESULT_ARENA = null;
const MAX_POOLED_BUFFERS_PER_SIZE = 16;
const RETAINED_BUFFERS = new Set();

function createResultArena() {
  // issued: handed out during the current iteration.
  // carried: handed out earlier and still reachable from runtime state. They are
  // re-checked every iteration, so the previous weight buffer returns to the
  // pool one iteration after 값 바꾸기 replaces it.
  return { issued: [], carried: [], free: new Map() };
}

// Contents are undefined: every caller must write all `length` elements.
function takeResultBuffer(length) {
  const arena = RESULT_ARENA;
  if (!arena) return new Float32Array(length);

  const pool = arena.free.get(length);
  const buffer = pool && pool.length ? pool.pop() : new Float32Array(length);
  arena.issued.push(buffer);
  return buffer;
}

function releaseResultBuffer(arena, buffer) {
  let pool = arena.free.get(buffer.length);
  if (!pool) {
    pool = [];
    arena.free.set(buffer.length, pool);
  }
  if (pool.length < MAX_POOLED_BUFFERS_PER_SIZE) pool.push(buffer);
}

function markRetainedBuffer(value) {
  if (isArrayValue(value)) RETAINED_BUFFERS.add(value.data);
}

function endResultArenaIteration(iterationResult) {
  const arena = RESULT_ARENA;
  if (!arena || (!arena.issued.length && !arena.carried.length)) return;

  RETAINED_BUFFERS.clear();
  markRetainedBuffer(iterationResult);
  for (const entry of RUNTIME_VARIABLES.values()) markRetainedBuffer(entry.value);
  if (pendingVariableUpdates) {
    for (const entry of pendingVariableUpdates.values()) markRetainedBuffer(entry.value);
  }

  const stillLive = [];
  for (const list of [arena.issued, arena.carried]) {
    for (let i = 0; i < list.length; i++) {
      const buffer = list[i];
      if (RETAINED_BUFFERS.has(buffer)) stillLive.push(buffer);
      else releaseResultBuffer(arena, buffer);
    }
    list.length = 0;
  }

  arena.carried = stillLive;
  RETAINED_BUFFERS.clear();
}

// Run one iteration of a compiled plan with the arena active. The arena is only
// installed for the synchronous body so that anything running during a later
// await — a UI action, another evaluation — never draws from it.
function withResultArena(arena, run) {
  const previous = RESULT_ARENA;
  RESULT_ARENA = arena;
  try {
    const result = run();
    endResultArenaIteration(result);
    return result;
  } catch (error) {
    // Values from a failed iteration may be referenced from anywhere; drop the
    // pool rather than reason about a half-finished step.
    arena.issued.length = 0;
    arena.carried.length = 0;
    arena.free.clear();
    throw error;
  } finally {
    RESULT_ARENA = previous;
  }
}

// ---------- 일반 원소별 연산 ----------
// Used by user-defined blocks and by any operation without a dedicated kernel.

function elementwiseUnary(value, fn) {
  if (typeof value === 'number') return fn(value);
  const arr = asArrayValue(value);
  const source = arr.data;
  const data = takeResultBuffer(source.length);
  for (let i = 0; i < data.length; i++) data[i] = fn(source[i], i);
  return fastArrayValue(data, arr.shape);
}

function elementwiseBinary(a, b, fn) {
  if (typeof a === 'number' && typeof b === 'number') return fn(a, b);
  // Each mixed case gets its own loop. Wrapping fn in another closure here used
  // to add a second layer of indirect calls to every element.
  if (typeof a === 'number') {
    const bb = asArrayValue(b), bd = bb.data;
    const data = takeResultBuffer(bd.length);
    for (let i = 0; i < data.length; i++) data[i] = fn(a, bd[i]);
    return fastArrayValue(data, bb.shape);
  }
  const aa = asArrayValue(a), ad = aa.data;
  if (typeof b === 'number') {
    const data = takeResultBuffer(ad.length);
    for (let i = 0; i < data.length; i++) data[i] = fn(ad[i], b);
    return fastArrayValue(data, aa.shape);
  }
  const bd = asArrayValue(b).data;
  if (ad.length !== bd.length) throw new Error('배열의 원소 수가 서로 다릅니다.');
  const data = takeResultBuffer(ad.length);
  for (let i = 0; i < data.length; i++) data[i] = fn(ad[i], bd[i]);
  return fastArrayValue(data, aa.shape);
}

// ---------- 원소별 연산 커널 ----------
// These are the innermost loops of both the forward pass and backpropagation. A
// single SGD step on a 15x784 weight matrix runs tens of thousands of elements
// through them, so they are written as plain typed loops instead of going
// through elementwiseBinary/elementwiseUnary with a per-element callback.

function addValues(a, b) {
  if (typeof a === 'number') {
    if (typeof b === 'number') return a + b;
    const bb = asArrayValue(b), bd = bb.data;
    const out = takeResultBuffer(bd.length);
    for (let i = 0; i < out.length; i++) out[i] = a + bd[i];
    return fastArrayValue(out, bb.shape);
  }
  const aa = asArrayValue(a), ad = aa.data;
  if (typeof b === 'number') {
    const out = takeResultBuffer(ad.length);
    for (let i = 0; i < out.length; i++) out[i] = ad[i] + b;
    return fastArrayValue(out, aa.shape);
  }
  const bd = asArrayValue(b).data;
  if (ad.length !== bd.length) throw new Error('배열의 원소 수가 서로 다릅니다.');
  const out = takeResultBuffer(ad.length);
  for (let i = 0; i < out.length; i++) out[i] = ad[i] + bd[i];
  return fastArrayValue(out, aa.shape);
}

function subtractValues(a, b) {
  if (typeof a === 'number') {
    if (typeof b === 'number') return a - b;
    const bb = asArrayValue(b), bd = bb.data;
    const out = takeResultBuffer(bd.length);
    for (let i = 0; i < out.length; i++) out[i] = a - bd[i];
    return fastArrayValue(out, bb.shape);
  }
  const aa = asArrayValue(a), ad = aa.data;
  if (typeof b === 'number') {
    const out = takeResultBuffer(ad.length);
    for (let i = 0; i < out.length; i++) out[i] = ad[i] - b;
    return fastArrayValue(out, aa.shape);
  }
  const bd = asArrayValue(b).data;
  if (ad.length !== bd.length) throw new Error('배열의 원소 수가 서로 다릅니다.');
  const out = takeResultBuffer(ad.length);
  for (let i = 0; i < out.length; i++) out[i] = ad[i] - bd[i];
  return fastArrayValue(out, aa.shape);
}

function multiplyValues(a, b) {
  if (typeof a === 'number') {
    if (typeof b === 'number') return a * b;
    const bb = asArrayValue(b), bd = bb.data;
    const out = takeResultBuffer(bd.length);
    for (let i = 0; i < out.length; i++) out[i] = a * bd[i];
    return fastArrayValue(out, bb.shape);
  }
  const aa = asArrayValue(a), ad = aa.data;
  if (typeof b === 'number') {
    const out = takeResultBuffer(ad.length);
    for (let i = 0; i < out.length; i++) out[i] = ad[i] * b;
    return fastArrayValue(out, aa.shape);
  }
  const bd = asArrayValue(b).data;
  if (ad.length !== bd.length) throw new Error('배열의 원소 수가 서로 다릅니다.');
  const out = takeResultBuffer(ad.length);
  for (let i = 0; i < out.length; i++) out[i] = ad[i] * bd[i];
  return fastArrayValue(out, aa.shape);
}

function divideValues(a, b) {
  if (typeof a === 'number') {
    if (typeof b === 'number') return a / b;
    const bb = asArrayValue(b), bd = bb.data;
    const out = takeResultBuffer(bd.length);
    for (let i = 0; i < out.length; i++) out[i] = a / bd[i];
    return fastArrayValue(out, bb.shape);
  }
  const aa = asArrayValue(a), ad = aa.data;
  if (typeof b === 'number') {
    const out = takeResultBuffer(ad.length);
    for (let i = 0; i < out.length; i++) out[i] = ad[i] / b;
    return fastArrayValue(out, aa.shape);
  }
  const bd = asArrayValue(b).data;
  if (ad.length !== bd.length) throw new Error('배열의 원소 수가 서로 다릅니다.');
  const out = takeResultBuffer(ad.length);
  for (let i = 0; i < out.length; i++) out[i] = ad[i] / bd[i];
  return fastArrayValue(out, aa.shape);
}

function maximumValues(a, b) {
  if (typeof a === 'number') {
    if (typeof b === 'number') return Math.max(a, b);
    const bb = asArrayValue(b), bd = bb.data;
    const out = takeResultBuffer(bd.length);
    for (let i = 0; i < out.length; i++) out[i] = Math.max(a, bd[i]);
    return fastArrayValue(out, bb.shape);
  }
  const aa = asArrayValue(a), ad = aa.data;
  if (typeof b === 'number') {
    const out = takeResultBuffer(ad.length);
    for (let i = 0; i < out.length; i++) out[i] = Math.max(ad[i], b);
    return fastArrayValue(out, aa.shape);
  }
  const bd = asArrayValue(b).data;
  if (ad.length !== bd.length) throw new Error('배열의 원소 수가 서로 다릅니다.');
  const out = takeResultBuffer(ad.length);
  for (let i = 0; i < out.length; i++) out[i] = Math.max(ad[i], bd[i]);
  return fastArrayValue(out, aa.shape);
}

function equalValues(a, b) {
  if (typeof a === 'number') {
    if (typeof b === 'number') return a === b ? 1 : 0;
    const bb = asArrayValue(b), bd = bb.data;
    const out = takeResultBuffer(bd.length);
    for (let i = 0; i < out.length; i++) out[i] = a === bd[i] ? 1 : 0;
    return fastArrayValue(out, bb.shape);
  }
  const aa = asArrayValue(a), ad = aa.data;
  if (typeof b === 'number') {
    const out = takeResultBuffer(ad.length);
    for (let i = 0; i < out.length; i++) out[i] = ad[i] === b ? 1 : 0;
    return fastArrayValue(out, aa.shape);
  }
  const bd = asArrayValue(b).data;
  if (ad.length !== bd.length) throw new Error('배열의 원소 수가 서로 다릅니다.');
  const out = takeResultBuffer(ad.length);
  for (let i = 0; i < out.length; i++) out[i] = ad[i] === bd[i] ? 1 : 0;
  return fastArrayValue(out, aa.shape);
}

function negateValue(v) {
  if (typeof v === 'number') return -v;
  const a = asArrayValue(v), d = a.data;
  const out = takeResultBuffer(d.length);
  for (let i = 0; i < out.length; i++) out[i] = -d[i];
  return fastArrayValue(out, a.shape);
}

function expValues(v) {
  if (typeof v === 'number') return Math.exp(v);
  const a = asArrayValue(v), d = a.data;
  const out = takeResultBuffer(d.length);
  for (let i = 0; i < out.length; i++) out[i] = Math.exp(d[i]);
  return fastArrayValue(out, a.shape);
}

function logValues(v) {
  if (typeof v === 'number') return Math.log(v);
  const a = asArrayValue(v), d = a.data;
  const out = takeResultBuffer(d.length);
  for (let i = 0; i < out.length; i++) out[i] = Math.log(d[i]);
  return fastArrayValue(out, a.shape);
}

function sumArray(arr) {
  const data = arr.data;
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += data[i];
  return sum;
}

function dotValues(a, b) {
  const aa = asArrayValue(a), bb = asArrayValue(b);
  if (aa.data.length !== bb.data.length) throw new Error('내적할 두 벡터의 길이가 다릅니다.');
  let s = 0;
  for (let i = 0; i < aa.data.length; i++) s += aa.data[i] * bb.data[i];
  return s;
}

function argmaxValue(value) {
  const array = asArrayValue(value);
  if (array.shape.length !== 1) throw new Error('분류 점수는 벡터여야 합니다.');
  if (!array.data.length) throw new Error('빈 벡터에서는 가장 큰 점수를 찾을 수 없습니다.');
  let bestIndex = 0;
  let bestValue = array.data[0];
  for (let i = 1; i < array.data.length; i++) {
    if (array.data[i] > bestValue) {
      bestValue = array.data[i];
      bestIndex = i;
    }
  }
  return bestIndex;
}

function arrayMaximumValue(value) {
  const array = asArrayValue(value);
  if (!array.data.length) throw new Error('빈 배열에서는 최댓값을 구할 수 없습니다.');
  let maximum = array.data[0];
  for (let i = 1; i < array.data.length; i++) maximum = Math.max(maximum, array.data[i]);
  return maximum;
}

// ---------- gradient shape helpers ----------

function zerosLike(v) {
  if (typeof v === 'number') return 0;
  if (isArrayValue(v)) {
    // A pooled buffer still holds the previous iteration's numbers, so zeros
    // have to be written explicitly rather than relying on a fresh allocation.
    const data = takeResultBuffer(v.data.length);
    data.fill(0);
    return fastArrayValue(data, v.shape);
  }
  return null;
}

function fillLike(v, scalar) {
  if (typeof v === 'number') return scalar;
  const a = asArrayValue(v);
  const data = takeResultBuffer(a.data.length);
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
  // Values are immutable by rule, so the first gradient for a node can be stored
  // as-is; addValues allocates a fresh array for every later accumulation.
  map.set(key, map.has(key) ? addValues(map.get(key), grad) : grad);
}

// ---------- random ----------

function deterministicRandomVector(length, seed) {
  let state = (Number(seed) || 1) >>> 0;
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    state = (1664525 * state + 1013904223) >>> 0;
    out[i] = (state / 4294967296) * 2 - 1;
  }
  return arrayValue(out, [length]);
}

function randomArray(shape, seed, scale = 1) {
  const length = shape.reduce((a, b) => a * b, 1);
  let state = (Number(seed) || 1) >>> 0;
  const out = new Float32Array(length);
  const s = Number(scale);
  for (let i = 0; i < length; i++) {
    state = (1664525 * state + 1013904223) >>> 0;
    out[i] = ((state / 4294967296) * 2 - 1) * s;
  }
  return arrayValue(out, shape);
}

// ---------- 행렬 커널 ----------

function matrixVectorValues(matrix, vector) {
  const m = asArrayValue(matrix);
  const v = asArrayValue(vector);
  if (m.shape.length !== 2) throw new Error('첫 번째 입력은 행렬이어야 합니다.');
  if (v.shape.length !== 1) throw new Error('두 번째 입력은 벡터여야 합니다.');
  const rows = m.shape[0];
  const cols = m.shape[1];
  const md = m.data;
  const vd = v.data;
  if (vd.length !== cols) throw new Error(`행렬의 열 ${cols}개와 벡터 길이 ${vd.length}가 다릅니다.`);

  const out = takeResultBuffer(rows);
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let sum = 0;
    let c = 0;
    for (; c + 7 < cols; c += 8) {
      sum += md[base + c] * vd[c];
      sum += md[base + c + 1] * vd[c + 1];
      sum += md[base + c + 2] * vd[c + 2];
      sum += md[base + c + 3] * vd[c + 3];
      sum += md[base + c + 4] * vd[c + 4];
      sum += md[base + c + 5] * vd[c + 5];
      sum += md[base + c + 6] * vd[c + 6];
      sum += md[base + c + 7] * vd[c + 7];
    }
    for (; c < cols; c++) sum += md[base + c] * vd[c];
    out[r] = sum;
  }
  return arrayValue(out, [rows]);
}

function transposeMatrixVectorValues(matrix, vector) {
  const m = asArrayValue(matrix);
  const v = asArrayValue(vector);
  if (m.shape.length !== 2 || v.shape.length !== 1) throw new Error('전치 행렬 계산의 모양이 맞지 않습니다.');
  const rows = m.shape[0];
  const cols = m.shape[1];
  const md = m.data;
  const vd = v.data;
  if (vd.length !== rows) throw new Error('전치 행렬과 벡터의 크기가 맞지 않습니다.');

  const out = takeResultBuffer(cols);
  // Read four adjacent columns together, retaining the original row-order
  // Float64 accumulation and the single Float32 store for each output.
  let c = 0;
  for (; c + 3 < cols; c += 4) {
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (let r = 0; r < rows; r++) {
      const base = r * cols + c, scale = vd[r];
      s0 += md[base] * scale;
      s1 += md[base + 1] * scale;
      s2 += md[base + 2] * scale;
      s3 += md[base + 3] * scale;
    }
    out[c] = s0; out[c + 1] = s1; out[c + 2] = s2; out[c + 3] = s3;
  }
  for (; c < cols; c++) {
    let sum = 0;
    let r = 0;
    for (; r + 7 < rows; r += 8) {
      sum += md[r * cols + c] * vd[r];
      sum += md[(r + 1) * cols + c] * vd[r + 1];
      sum += md[(r + 2) * cols + c] * vd[r + 2];
      sum += md[(r + 3) * cols + c] * vd[r + 3];
      sum += md[(r + 4) * cols + c] * vd[r + 4];
      sum += md[(r + 5) * cols + c] * vd[r + 5];
      sum += md[(r + 6) * cols + c] * vd[r + 6];
      sum += md[(r + 7) * cols + c] * vd[r + 7];
    }
    for (; r < rows; r++) sum += md[r * cols + c] * vd[r];
    out[c] = sum;
  }
  return arrayValue(out, [cols]);
}

function outerProductValues(a, b) {
  const aa = asArrayValue(a);
  const bb = asArrayValue(b);
  if (aa.shape.length !== 1 || bb.shape.length !== 1) throw new Error('외적에는 벡터 두 개가 필요합니다.');
  const ad = aa.data;
  const bd = bb.data;
  const cols = bd.length;
  const out = takeResultBuffer(ad.length * cols);
  for (let r = 0; r < ad.length; r++) {
    const scale = ad[r];
    const base = r * cols;
    for (let c = 0; c < cols; c++) out[base + c] = scale * bd[c];
  }
  return arrayValue(out, [ad.length, cols]);
}

// The three matrix kernels dominate training time, so the profiler measures them
// individually. Decorating them here — once, next to their definitions — keeps
// the measurement out of the modules that merely call them.
function timedKernel(name, kernel) {
  return function(...args) {
    const hook = PROFILER_HOOKS.kernel;
    if (!hook || !hook.active()) return kernel(...args);
    const started = performance.now();
    try {
      return kernel(...args);
    } finally {
      hook.record(name, performance.now() - started);
    }
  };
}

matrixVectorValues = timedKernel('행렬 × 벡터', matrixVectorValues);
transposeMatrixVectorValues = timedKernel('전치 행렬 × 벡터', transposeMatrixVectorValues);
outerProductValues = timedKernel('외적', outerProductValues);
