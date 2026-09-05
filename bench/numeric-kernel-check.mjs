// Bit-level regressions for optimized math; no browser or packages required.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const root = new URL('../', import.meta.url);
globalThis.PROFILER_HOOKS = {};
for (const file of ['js/values.js', 'js/spatial-ops.js']) {
  vm.runInThisContext(fs.readFileSync(new URL(file, root), 'utf8'), { filename: file });
}
vm.runInThisContext(fs.readFileSync(new URL('js/manual-math-extensions.js', root), 'utf8').split('BLOCKS.fold2d =')[0]);

function exact(actual, expected, label) {
  assert.deepEqual(actual.shape, expected.shape, label + ' shape');
  assert.deepEqual(Buffer.from(actual.data.buffer, actual.data.byteOffset, actual.data.byteLength),
    Buffer.from(expected.data.buffer, expected.data.byteOffset, expected.data.byteLength), label);
}

for (const [rows, cols] of [[1, 1], [1, 9], [3, 2], [7, 7], [15, 32], [32, 64], [64, 169], [64, 784], [676, 9]]) {
  for (const seed of [1, 19, 421]) {
    const matrix = randomArray([rows, cols], seed, 10), vector = randomArray([rows], seed + 1, 10);
    const reference = new Float32Array(cols);
    for (let c = 0; c < cols; c++) {
      let sum = 0;
      for (let r = 0; r < rows; r++) sum += matrix.data[r * cols + c] * vector.data[r];
      reference[c] = sum;
    }
    exact(transposeMatrixVectorValues(matrix, vector), arrayValue(reference, [cols]), `transpose ${rows}x${cols}`);
  }
}
assert.throws(() => transposeMatrixVectorValues(arrayValue([1, 2], [1, 2]), arrayValue([1, 2], [2])));

// One node is deliberately reused across geometry changes, including equal-size
// arrays with different shapes. Cached offsets must never survive those changes.
const node = { params: {} };
for (const [h, w, kh, kw, sh, sw, pad] of [
  [28, 28, 3, 3, 1, 1, 0], [28, 28, 3, 3, 1, 1, 1],
  [26, 26, 2, 2, 2, 2, 0], [4, 6, 2, 3, 1, 2, 1],
  [6, 4, 3, 2, 2, 1, 1], [3, 4, 1, 1, 1, 1, 0],
  [1, 1, 3, 3, 1, 1, 2], [40, 40, 26, 26, 1, 1, 20]
]) {
  Object.assign(node.params, { kernelRows: kh, kernelCols: kw, strideRows: sh, strideCols: sw, padding: pad });
  const input = randomArray([h, w], 7, 1);
  const nr = Math.floor((h + 2 * pad - kh) / sh) + 1, nc = Math.floor((w + 2 * pad - kw) / sw) + 1;
  const expected = new Float32Array(nr * nc * kh * kw);
  const patches = randomArray([nr * nc, kh * kw], 97, 1000);
  const folded = new Float32Array(h * w);
  let i = 0;
  for (let y = 0; y < nr; y++) for (let x = 0; x < nc; x++) {
    for (let ky = 0; ky < kh; ky++) for (let kx = 0; kx < kw; kx++, i++) {
      const sy = y * sh + ky - pad, sx = x * sw + kx - pad;
      if (sy < 0 || sy >= h || sx < 0 || sx >= w) continue;
      expected[i] = input.data[sy * w + sx];
      folded[sy * w + sx] += patches.data[i];
    }
  }
  for (let repeat = 0; repeat < 2; repeat++) {
    exact(unfold2dValues(node, input), arrayValue(expected, [nr * nc, kh * kw]), 'unfold');
    exact(fold2dValues(node, patches, input), arrayValue(folded, [h, w]), 'fold');
  }
}

console.log('PASS: transpose and spatial cache invalidation/fallback');
