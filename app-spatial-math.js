// General-purpose spatial array transforms for building convolution-like systems
// from math blocks. These are intentionally not AI-layer blocks: reshape only
// changes array shape, and sliding-window unfold extracts local patches.

(function installSpatialMathBlocks() {
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

  function unfoldConfig(node, input) {
    const arr = asArrayValue(input);
    if (arr.shape.length !== 2) {
      throw new Error('슬라이딩 창 펼치기에는 2차원 배열이 필요합니다.');
    }

    const [height, width] = arr.shape;
    const kernelRows = positiveInt(node.params.kernelRows, 3);
    const kernelCols = positiveInt(node.params.kernelCols, 3);
    const strideRows = positiveInt(node.params.strideRows, 1);
    const strideCols = positiveInt(node.params.strideCols, 1);
    const padding = nonNegativeInt(node.params.padding, 0);

    const paddedHeight = height + padding * 2;
    const paddedWidth = width + padding * 2;
    if (kernelRows > paddedHeight || kernelCols > paddedWidth) {
      throw new Error('슬라이딩 창이 패딩을 포함한 입력보다 큽니다.');
    }

    const outRows = Math.floor((paddedHeight - kernelRows) / strideRows) + 1;
    const outCols = Math.floor((paddedWidth - kernelCols) / strideCols) + 1;
    if (outRows < 1 || outCols < 1) throw new Error('슬라이딩 창의 출력 크기가 0입니다.');

    return {
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
  }

  function unfold2dValues(node, input) {
    const arr = asArrayValue(input);
    const config = unfoldConfig(node, arr);
    const out = takeResultBuffer(config.windowCount * config.windowSize);
    const src = arr.data;
    let write = 0;

    for (let outRow = 0; outRow < config.outRows; outRow++) {
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
    // primitiveVJP receives the direct node output, so keeping the extraction
    // geometry here lets backward use the exact parameters without changing the
    // global VJP call signature.
    value.unfold2d = config;
    return value;
  }

  function reshapeBackward(input, upstream) {
    const original = asArrayValue(input);
    const gradient = asArrayValue(upstream);
    if (gradient.data.length !== original.data.length) {
      throw new Error('배열 모양 바꾸기 역전파의 원소 수가 맞지 않습니다.');
    }
    return fastArrayValue(gradient.data, original.shape);
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

  BLOCKS.reshape = {
    title: '배열 모양 바꾸기',
    kind: 'transform',
    inputs: ['x'],
    description: '원소 순서는 그대로 두고 배열의 모양만 바꾼다. -1을 한 번 쓰면 나머지 크기를 자동 계산한다.',
    formula: node => `reshape(x, ${String(node.params.shape || '28,28').replace(/,/g, '×')})`,
    controls: [
      { key: 'shape', label: '새 모양 (예: 26,26 또는 -1,9)', type: 'text', default: '28,28' }
    ],
    compute: (node, [input]) => reshapeValues(input, node.params.shape)
  };

  BLOCKS.unfold2d = {
    title: '슬라이딩 창 펼치기',
    kind: 'transform',
    inputs: ['2차원 배열'],
    description: '2차원 배열에서 일정한 크기의 창을 움직이며 각 부분을 한 줄로 펼쳐 행렬로 만든다. 합성곱과 지역 연산을 일반 행렬 연산으로 표현할 수 있다.',
    formula: node => `unfold(x, ${node.params.kernelRows}×${node.params.kernelCols}, stride=${node.params.strideRows}×${node.params.strideCols}, pad=${node.params.padding})`,
    controls: [
      { key: 'kernelRows', label: '창 높이', type: 'number', step: '1', default: 3 },
      { key: 'kernelCols', label: '창 너비', type: 'number', step: '1', default: 3 },
      { key: 'strideRows', label: '세로 간격', type: 'number', step: '1', default: 1 },
      { key: 'strideCols', label: '가로 간격', type: 'number', step: '1', default: 1 },
      { key: 'padding', label: '바깥 0 채우기', type: 'number', step: '1', default: 0 }
    ],
    compute: (node, [input]) => unfold2dValues(node, input)
  };

  // Add both blocks beside the existing general transforms before app-boot.js
  // installs palette click handlers.
  const flattenButton = document.querySelector('[data-block="flatten"]');
  const transformSection = flattenButton?.closest('.palette-group');
  if (transformSection && !transformSection.querySelector('[data-block="reshape"]')) {
    const reshapeButton = document.createElement('button');
    reshapeButton.className = 'palette-block transform';
    reshapeButton.dataset.block = 'reshape';
    reshapeButton.textContent = '배열 모양 바꾸기';

    const unfoldButton = document.createElement('button');
    unfoldButton.className = 'palette-block transform';
    unfoldButton.dataset.block = 'unfold2d';
    unfoldButton.textContent = '슬라이딩 창 펼치기';

    transformSection.appendChild(reshapeButton);
    transformSection.appendChild(unfoldButton);
  }

  const primitiveVJPBeforeSpatial = primitiveVJP;
  primitiveVJP = function(type, inputs, output, upstream) {
    if (type === 'reshape') return [reshapeBackward(inputs[0], upstream)];
    if (type === 'unfold2d') return [unfold2dBackward(inputs[0], output, upstream)];
    return primitiveVJPBeforeSpatial(type, inputs, output, upstream);
  };
})();
