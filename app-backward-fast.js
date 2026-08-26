// Fast kernels for the remaining reverse-mode bottlenecks seen in the CNN profiler.
// Visible blocks and graph semantics are unchanged.

(function installBackwardFastPaths() {
  const transposeMatrixVectorValuesBeforeBackwardFast = transposeMatrixVectorValues;
  const outerProductValuesBeforeBackwardFast = outerProductValues;
  const primitiveVJPBeforeBackwardFast = primitiveVJP;

  function useFastPath() {
    return window.BACKWARD_FAST_DISABLED !== true;
  }

  // Same summation order per output column as the original implementation, but
  // process eight adjacent columns together so matrix reads are contiguous.
  transposeMatrixVectorValues = function(matrix, vector) {
    if (!useFastPath()) return transposeMatrixVectorValuesBeforeBackwardFast(matrix, vector);

    const m = asArrayValue(matrix);
    const v = asArrayValue(vector);
    if (m.shape.length !== 2 || v.shape.length !== 1) {
      throw new Error('전치 행렬 계산의 모양이 맞지 않습니다.');
    }

    const rows = m.shape[0];
    const cols = m.shape[1];
    if (v.data.length !== rows) throw new Error('전치 행렬과 벡터의 크기가 맞지 않습니다.');

    const md = m.data;
    const vd = v.data;
    const out = takeResultBuffer(cols);
    let c = 0;

    for (; c + 7 < cols; c += 8) {
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      let s4 = 0, s5 = 0, s6 = 0, s7 = 0;

      for (let r = 0; r < rows; r++) {
        const base = r * cols + c;
        const scale = vd[r];
        s0 += md[base] * scale;
        s1 += md[base + 1] * scale;
        s2 += md[base + 2] * scale;
        s3 += md[base + 3] * scale;
        s4 += md[base + 4] * scale;
        s5 += md[base + 5] * scale;
        s6 += md[base + 6] * scale;
        s7 += md[base + 7] * scale;
      }

      out[c] = s0;
      out[c + 1] = s1;
      out[c + 2] = s2;
      out[c + 3] = s3;
      out[c + 4] = s4;
      out[c + 5] = s5;
      out[c + 6] = s6;
      out[c + 7] = s7;
    }

    for (; c < cols; c++) {
      let sum = 0;
      for (let r = 0; r < rows; r++) sum += md[r * cols + c] * vd[r];
      out[c] = sum;
    }

    return fastArrayValue(out, [cols]);
  };

  // Outer product has no reduction, so eight independent outputs can be emitted
  // per loop without changing floating-point results.
  outerProductValues = function(a, b) {
    if (!useFastPath()) return outerProductValuesBeforeBackwardFast(a, b);

    const aa = asArrayValue(a);
    const bb = asArrayValue(b);
    if (aa.shape.length !== 1 || bb.shape.length !== 1) {
      throw new Error('외적에는 벡터 두 개가 필요합니다.');
    }

    const ad = aa.data;
    const bd = bb.data;
    const cols = bd.length;
    const out = takeResultBuffer(ad.length * cols);

    for (let r = 0; r < ad.length; r++) {
      const scale = ad[r];
      const base = r * cols;
      let c = 0;
      for (; c + 7 < cols; c += 8) {
        out[base + c] = scale * bd[c];
        out[base + c + 1] = scale * bd[c + 1];
        out[base + c + 2] = scale * bd[c + 2];
        out[base + c + 3] = scale * bd[c + 3];
        out[base + c + 4] = scale * bd[c + 4];
        out[base + c + 5] = scale * bd[c + 5];
        out[base + c + 6] = scale * bd[c + 6];
        out[base + c + 7] = scale * bd[c + 7];
      }
      for (; c < cols; c++) out[base + c] = scale * bd[c];
    }

    return fastArrayValue(out, [ad.length, cols]);
  };

  function upstreamReader(upstream, length) {
    if (typeof upstream === 'number') {
      return { scalar: true, value: upstream, data: null };
    }
    const arr = asArrayValue(upstream);
    if (arr.data.length !== length) throw new Error('최댓값 역전파의 gradient 크기가 맞지 않습니다.');
    return { scalar: false, value: 0, data: arr.data };
  }

  // ReLU in this project is expressed as maximum(0, x). The generic VJP used to
  // allocate two masks and two multiplied arrays, then reduce the scalar side.
  // Mixed scalar/array maximum can produce the useful array gradient and scalar
  // reduction in one pass. Math.fround reproduces the Float32 intermediate that
  // the old multiplyValues buffer introduced before unbroadcast summed it.
  function maximumMixedVJP(a, b, upstream) {
    if (typeof a === 'number' && isArrayValue(b)) {
      const bb = asArrayValue(b);
      const read = upstreamReader(upstream, bb.data.length);
      const gradB = takeResultBuffer(bb.data.length);
      let gradA = 0;

      for (let i = 0; i < bb.data.length; i++) {
        const x = bb.data[i];
        const u = read.scalar ? read.value : read.data[i];
        let fa, fb;
        if (a > x) { fa = 1; fb = 0; }
        else if (a < x) { fa = 0; fb = 1; }
        else { fa = 0.5; fb = 0.5; }

        gradA += Math.fround(u * fa);
        gradB[i] = u * fb;
      }

      return [gradA, fastArrayValue(gradB, bb.shape)];
    }

    if (isArrayValue(a) && typeof b === 'number') {
      const aa = asArrayValue(a);
      const read = upstreamReader(upstream, aa.data.length);
      const gradA = takeResultBuffer(aa.data.length);
      let gradB = 0;

      for (let i = 0; i < aa.data.length; i++) {
        const x = aa.data[i];
        const u = read.scalar ? read.value : read.data[i];
        let fa, fb;
        if (x > b) { fa = 1; fb = 0; }
        else if (x < b) { fa = 0; fb = 1; }
        else { fa = 0.5; fb = 0.5; }

        gradA[i] = u * fa;
        gradB += Math.fround(u * fb);
      }

      return [fastArrayValue(gradA, aa.shape), gradB];
    }

    return null;
  }

  primitiveVJP = function(type, inputs, output, upstream) {
    if (useFastPath() && type === 'maximum') {
      const mixed = maximumMixedVJP(inputs[0], inputs[1], upstream);
      if (mixed) return mixed;
    }
    return primitiveVJPBeforeBackwardFast(type, inputs, output, upstream);
  };

  window.BACKWARD_FAST = {
    setDisabled(value) { window.BACKWARD_FAST_DISABLED = Boolean(value); },
    get disabled() { return window.BACKWARD_FAST_DISABLED === true; }
  };
})();
