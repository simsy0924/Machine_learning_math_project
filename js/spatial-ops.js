// General-purpose spatial array transforms.
//
// These are intentionally not AI-layer operations: reshape only changes an
// array's shape, and sliding-window unfold extracts local patches. Convolution
// is then just 슬라이딩 창 펼치기 → 행렬 × 벡터, built from ordinary blocks.
//
// Forward and user-authored backward transforms share cached window geometry.
const UNFOLD_CONFIG_CACHE = new WeakMap();
const MAX_CACHED_WINDOW_INDICES = 1_000_000;

function positiveInt(value, fallback = 1) {
  const n = Math.floor(Number(value));
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function nonNegativeInt(value, fallback = 0) {
  const n = Math.floor(Number(value));
  return Number.isFinite(n) && n >= 0 ? n : fallback;
}

function parseReshapeSpec(raw, length) {
  const text = String(raw ?? '').trim();
  if (!text) throw new Error('새 배열 모양을 입력하세요. 예: 26,26');

  const parts = text
    .split(/[x×,\s]+/)
    .map(part => part.trim())
    .filter(Boolean);

  if (!parts.length) throw new Error('새 배열 모양을 입력하세요. 예: 26,26');

  const shape = parts.map(part => {
    const value = Number(part);
    if (!Number.isInteger(value) || value === 0 || value < -1) {
      throw new Error(`배열 크기 '${part}'는 양의 정수 또는 -1이어야 합니다.`);
    }
    return value;
  });

  const inferIndex = shape.indexOf(-1);
  if (inferIndex >= 0 && shape.lastIndexOf(-1) !== inferIndex) {
    throw new Error('자동 계산 크기 -1은 한 번만 사용할 수 있습니다.');
  }

  let knownProduct = 1;
  for (const dim of shape) if (dim !== -1) knownProduct *= dim;

  if (inferIndex >= 0) {
    if (length % knownProduct !== 0) {
      throw new Error(`원소 ${length}개를 요청한 모양으로 바꿀 수 없습니다.`);
    }
    shape[inferIndex] = length / knownProduct;
  }

  const product = shape.reduce((a, b) => a * b, 1);
  if (product !== length) {
    throw new Error(`원소 수가 맞지 않습니다. 입력 ${length}개, 새 모양 ${shape.join('×')} = ${product}개입니다.`);
  }
  return shape;
}

function reshapeValues(input, rawShape) {
  const arr = asArrayValue(input);
  const shape = parseReshapeSpec(rawShape, arr.data.length);
  // Array values are immutable by runtime rule, so reshape is a zero-copy view.
  return fastArrayValue(arr.data, shape);
}

function reshapeBackward(input, upstream) {
  const original = asArrayValue(input);
  const gradient = asArrayValue(upstream);
  if (gradient.data.length !== original.data.length) {
    throw new Error('배열 모양 바꾸기 역전파의 원소 수가 맞지 않습니다.');
  }
  return fastArrayValue(gradient.data, original.shape);
}

function unfoldConfig(node, input) {
  const arr = asArrayValue(input);
  if (arr.shape.length !== 2) {
    throw new Error('슬라이딩 창 펼치기에는 2차원 배열이 필요합니다.');
  }

  const height = arr.shape[0];
  const width = arr.shape[1];
  const kernelRows = positiveInt(node.params.kernelRows, 3);
  const kernelCols = positiveInt(node.params.kernelCols, 3);
  const strideRows = positiveInt(node.params.strideRows, 1);
  const strideCols = positiveInt(node.params.strideCols, 1);
  const padding = nonNegativeInt(node.params.padding, 0);

  const cached = UNFOLD_CONFIG_CACHE.get(node);
  if (cached && cached.height === height && cached.width === width
    && cached.kernelRows === kernelRows && cached.kernelCols === kernelCols
    && cached.strideRows === strideRows && cached.strideCols === strideCols
    && cached.padding === padding) return cached;

  const paddedHeight = height + padding * 2;
  const paddedWidth = width + padding * 2;
  if (kernelRows > paddedHeight || kernelCols > paddedWidth) {
    throw new Error('슬라이딩 창이 패딩을 포함한 입력보다 큽니다.');
  }

  const outRows = Math.floor((paddedHeight - kernelRows) / strideRows) + 1;
  const outCols = Math.floor((paddedWidth - kernelCols) / strideCols) + 1;
  if (outRows < 1 || outCols < 1) throw new Error('슬라이딩 창의 출력 크기가 0입니다.');

  const config = {
    height,
    width,
    kernelRows,
    kernelCols,
    strideRows,
    strideCols,
    padding,
    outRows,
    outCols,
    windowSize: kernelRows * kernelCols,
    windowCount: outRows * outCols
  };
  UNFOLD_CONFIG_CACHE.set(node, config);
  return config;
}

// Cache positions rather than pixel values. Editing geometry or changing the
// input shape produces a new config; very large windows retain the loop path.
function windowSourceIndices(config) {
  if (config.sourceIndices !== undefined) return config.sourceIndices;
  const length = config.windowCount * config.windowSize;
  if (length > MAX_CACHED_WINDOW_INDICES || config.height * config.width > 0x7fffffff) {
    config.sourceIndices = null;
    return null;
  }
  const indices = new Int32Array(length);
  let write = 0;
  for (let y = 0; y < config.outRows; y++) {
    const top = y * config.strideRows - config.padding;
    for (let x = 0; x < config.outCols; x++) {
      const left = x * config.strideCols - config.padding;
      for (let kr = 0; kr < config.kernelRows; kr++) {
        const row = top + kr;
        for (let kc = 0; kc < config.kernelCols; kc++) {
          const col = left + kc;
          indices[write++] = row >= 0 && row < config.height && col >= 0 && col < config.width
            ? row * config.width + col : -1;
        }
      }
    }
  }
  config.sourceIndices = indices;
  return indices;
}

function unfold2dValues(node, input) {
  const arr = asArrayValue(input);
  const config = unfoldConfig(node, arr);
  const out = takeResultBuffer(config.windowCount * config.windowSize);
  const src = arr.data;
  let write = 0;

  const indices = windowSourceIndices(config);
  if (indices) {
    for (let i = 0; i < indices.length; i++) out[i] = indices[i] < 0 ? 0 : src[indices[i]];
  } else for (let outRow = 0; outRow < config.outRows; outRow++) {
    const sourceTop = outRow * config.strideRows - config.padding;
    for (let outCol = 0; outCol < config.outCols; outCol++) {
      const sourceLeft = outCol * config.strideCols - config.padding;
      for (let kr = 0; kr < config.kernelRows; kr++) {
        const sourceRow = sourceTop + kr;
        for (let kc = 0; kc < config.kernelCols; kc++) {
          const sourceCol = sourceLeft + kc;
          out[write++] = sourceRow >= 0 && sourceRow < config.height && sourceCol >= 0 && sourceCol < config.width
            ? src[sourceRow * config.width + sourceCol]
            : 0;
        }
      }
    }
  }

  const value = fastArrayValue(out, [config.windowCount, config.windowSize]);
  // The VJP receives the direct node output, so keeping the extraction geometry
  // here lets backward reuse the exact parameters without a wider signature.
  value.unfold2d = config;
  return value;
}

function unfold2dBackward(input, output, upstream) {
  const original = asArrayValue(input);
  const gradient = asArrayValue(upstream);
  const config = output?.unfold2d;
  if (!config) throw new Error('슬라이딩 창 역전파 정보를 찾지 못했습니다.');
  if (gradient.data.length !== config.windowCount * config.windowSize) {
    throw new Error('슬라이딩 창 역전파의 gradient 크기가 맞지 않습니다.');
  }

  const out = takeResultBuffer(original.data.length);
  out.fill(0);
  const gd = gradient.data;
  let read = 0;

  for (let outRow = 0; outRow < config.outRows; outRow++) {
    const sourceTop = outRow * config.strideRows - config.padding;
    for (let outCol = 0; outCol < config.outCols; outCol++) {
      const sourceLeft = outCol * config.strideCols - config.padding;
      for (let kr = 0; kr < config.kernelRows; kr++) {
        const sourceRow = sourceTop + kr;
        for (let kc = 0; kc < config.kernelCols; kc++) {
          const sourceCol = sourceLeft + kc;
          const g = gd[read++];
          if (sourceRow >= 0 && sourceRow < config.height && sourceCol >= 0 && sourceCol < config.width) {
            out[sourceRow * config.width + sourceCol] += g;
          }
        }
      }
    }
  }

  return fastArrayValue(out, original.shape);
}
