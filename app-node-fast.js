// Node-level runtime optimizations for repeated math evaluation.
// These keep block semantics unchanged while removing avoidable copies and
// allocations from hot paths. Array values are treated as immutable throughout
// the runtime, so read-only buffers can be shared safely between view-like nodes.

(function installNodeRuntimeOptimizations() {
  const constantVectorCache = new WeakMap();
  const randomVectorCache = new WeakMap();
  const matrixCache = new WeakMap();
  const oneHotCache = new Map();
  let cachedDatasetClasses = [];
  let cachedDatasetValue = null;

  function cachedNodeValue(cache, node, signature, build) {
    const previous = cache.get(node);
    if (previous && previous.signature === signature) return previous.value;
    const value = build();
    cache.set(node, { signature, value });
    return value;
  }

  function currentDatasetValue() {
    const api = window.quickDrawDataset;
    if (!api) throw new Error('데이터 로더가 아직 준비되지 않았습니다.');
    const cache = api.cache;
    if (!cache || !cache.size) throw new Error('오른쪽에서 Quick Draw 데이터를 먼저 불러오세요.');

    let unchanged = cache.size === cachedDatasetClasses.length;
    if (unchanged) {
      let i = 0;
      for (const name of cache.keys()) {
        if (cachedDatasetClasses[i++] !== name) {
          unchanged = false;
          break;
        }
      }
    }

    if (!unchanged || !cachedDatasetValue) {
      cachedDatasetClasses = Array.from(cache.keys());
      cachedDatasetValue = { kind: 'dataset', classes: cachedDatasetClasses };
    }
    return cachedDatasetValue;
  }

  // Dataset metadata is unchanged during one long calculation. Reuse the same
  // immutable class list instead of allocating Array.from(...) every sample.
  BLOCKS.dataset.compute = () => currentDatasetValue();

  // A normalized sample image is already a Float32Array. "그림값" only adds a
  // 28x28 view, and "펼치기" only changes the shape; neither needs to copy 784
  // floating-point values.
  BLOCKS.sampleImage.compute = (node, [sample]) => {
    if (!sample || sample.kind !== 'sample') throw new Error('샘플 고르기 블록이 필요합니다.');
    return arrayValue(sample.image, [28, 28]);
  };

  BLOCKS.flatten.compute = (node, [input]) => {
    const arr = asArrayValue(input);
    return arrayValue(arr.data, [arr.data.length]);
  };

  // One-hot vectors depend only on class count and class index and are immutable.
  // Reuse the tiny vectors instead of allocating them for every sample.
  BLOCKS.sampleOneHot.compute = (node, [sample]) => {
    if (!sample || sample.kind !== 'sample') throw new Error('샘플 고르기 블록이 필요합니다.');
    const length = sample.classes.length;
    const key = `${length}:${sample.classIndex}`;
    let value = oneHotCache.get(key);
    if (!value) {
      const data = new Float32Array(length);
      data[sample.classIndex] = 1;
      value = arrayValue(data, [length]);
      oneHotCache.set(key, value);
    }
    return value;
  };

  // Constant/deterministic generator nodes are loop-invariant unless their
  // parameters are edited. Cache by node + parameter signature.
  BLOCKS.constantVector.compute = node => {
    const length = Math.max(1, Math.floor(Number(node.params.length) || 1));
    const scalar = Number(node.params.value);
    const signature = `${length}|${scalar}`;
    return cachedNodeValue(constantVectorCache, node, signature, () => {
      const data = new Float32Array(length);
      data.fill(scalar);
      return arrayValue(data, [length]);
    });
  };

  BLOCKS.randomVector.compute = node => {
    const length = Math.max(1, Math.floor(Number(node.params.length) || 1));
    const seed = Number(node.params.seed) || 1;
    const signature = `${length}|${seed}`;
    return cachedNodeValue(
      randomVectorCache,
      node,
      signature,
      () => deterministicRandomVector(length, seed)
    );
  };

  if (BLOCKS.matrix) {
    BLOCKS.matrix.compute = node => {
      const rows = Math.max(1, Math.floor(Number(node.params.rows) || 1));
      const cols = Math.max(1, Math.floor(Number(node.params.cols) || 1));
      const init = String(node.params.init || 'constant');
      const value = Number(node.params.value);
      const seed = Number(node.params.seed) || 1;
      const scale = Number(node.params.scale);
      const signature = `${rows}|${cols}|${init}|${value}|${seed}|${scale}`;
      return cachedNodeValue(matrixCache, node, signature, () => {
        if (init === 'random') return randomArray([rows, cols], seed, scale);
        const data = new Float32Array(rows * cols);
        data.fill(value);
        return arrayValue(data, [rows, cols]);
      });
    };
  }

  // The repeated dataset node still creates one sample object and one normalized
  // Float32Array per new image, but shares the immutable class list instead of
  // cloning it every time.
  if (BLOCKS.datasetSampleByIndex) {
    BLOCKS.datasetSampleByIndex.compute = (node, [dataset, indexValue]) => {
      if (!dataset || dataset.kind !== 'dataset') throw new Error('불러온 데이터 블록이 필요합니다.');
      if (typeof indexValue !== 'number') throw new Error('샘플 번호는 숫자여야 합니다.');
      const api = window.quickDrawDataset;
      const classes = dataset.classes;
      if (!classes.length) throw new Error('데이터 종류가 없습니다.');
      const i = Math.max(0, Math.floor(indexValue));
      const classIndex = i % classes.length;
      const sampleIndex = Math.floor(i / classes.length) % 10000;
      const className = classes[classIndex];
      const image = api.getNormalizedImage(className, sampleIndex);
      return { kind: 'sample', className, classIndex, classes, index: sampleIndex, image };
    };
  }

  // Indexed loops avoid iterator overhead in reductions.
  sumArray = function(arr) {
    const data = arr.data;
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i];
    return sum;
  };

  // Keep the exact sequential accumulation order, but unroll the inner loop to
  // reduce loop-control overhead in the dominant 32x784 matrix-vector product.
  matrixVectorValues = function(matrix, vector) {
    const m = asArrayValue(matrix);
    const v = asArrayValue(vector);
    if (m.shape.length !== 2) throw new Error('첫 번째 입력은 행렬이어야 합니다.');
    if (v.shape.length !== 1) throw new Error('두 번째 입력은 벡터여야 합니다.');
    const rows = m.shape[0];
    const cols = m.shape[1];
    const md = m.data;
    const vd = v.data;
    if (vd.length !== cols) throw new Error(`행렬의 열 ${cols}개와 벡터 길이 ${vd.length}가 다릅니다.`);

    const out = new Float32Array(rows);
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
  };

  transposeMatrixVectorValues = function(matrix, vector) {
    const m = asArrayValue(matrix);
    const v = asArrayValue(vector);
    if (m.shape.length !== 2 || v.shape.length !== 1) throw new Error('전치 행렬 계산의 모양이 맞지 않습니다.');
    const rows = m.shape[0];
    const cols = m.shape[1];
    const md = m.data;
    const vd = v.data;
    if (vd.length !== rows) throw new Error('전치 행렬과 벡터의 크기가 맞지 않습니다.');

    const out = new Float32Array(cols);
    for (let c = 0; c < cols; c++) {
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
  };

  outerProductValues = function(a, b) {
    const aa = asArrayValue(a);
    const bb = asArrayValue(b);
    if (aa.shape.length !== 1 || bb.shape.length !== 1) throw new Error('외적에는 벡터 두 개가 필요합니다.');
    const ad = aa.data;
    const bd = bb.data;
    const cols = bd.length;
    const out = new Float32Array(ad.length * cols);
    for (let r = 0; r < ad.length; r++) {
      const scale = ad[r];
      const base = r * cols;
      for (let c = 0; c < cols; c++) out[base + c] = scale * bd[c];
    }
    return arrayValue(out, [ad.length, cols]);
  };
})();
