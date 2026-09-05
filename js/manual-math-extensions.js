// General math primitives and small compatibility helpers for manual training.
//
// These are ordinary transforms, not automatic-differentiation operations.
// In particular, fold2d is the linear complement of sliding-window extraction:
// it places every patch element back at its source position and sums values
// where windows overlap.

function fold2dValues(node, patches, like) {
  const patchArray = asArrayValue(patches);
  const likeArray = asArrayValue(like);
  const config = unfoldConfig(node, likeArray);

  const expected = config.windowCount * config.windowSize;
  if (patchArray.data.length !== expected) {
    throw new Error(`슬라이딩 창 합치기 입력 크기가 맞지 않습니다. 필요 ${expected}개, 입력 ${patchArray.data.length}개입니다.`);
  }

  const out = takeResultBuffer(likeArray.data.length);
  out.fill(0);
  const source = patchArray.data;
  let read = 0;

  const indices = windowSourceIndices(config);
  if (indices) {
    // Same patch order and Float32 store at each addition as the loop path.
    for (let i = 0; i < indices.length; i++) {
      const target = indices[i];
      if (target >= 0) out[target] += source[i];
    }
  } else for (let outRow = 0; outRow < config.outRows; outRow++) {
    const targetTop = outRow * config.strideRows - config.padding;
    for (let outCol = 0; outCol < config.outCols; outCol++) {
      const targetLeft = outCol * config.strideCols - config.padding;
      for (let kr = 0; kr < config.kernelRows; kr++) {
        const targetRow = targetTop + kr;
        for (let kc = 0; kc < config.kernelCols; kc++) {
          const targetCol = targetLeft + kc;
          const value = source[read++];
          if (targetRow >= 0 && targetRow < config.height && targetCol >= 0 && targetCol < config.width) {
            out[targetRow * config.width + targetCol] += value;
          }
        }
      }
    }
  }

  return fastArrayValue(out, likeArray.shape);
}

BLOCKS.fold2d = {
  title: '슬라이딩 창 합치기',
  kind: 'transform',
  inputs: ['창 값', '원본 모양'],
  description: '펼쳐진 각 창의 값을 원래 2차원 위치로 되돌려 놓고, 겹치는 위치는 모두 더한다. 역전파뿐 아니라 일반적인 patch 집계 연산으로 사용할 수 있다.',
  formula: node => `fold2d(k=${node.params.kernelRows || 3}×${node.params.kernelCols || 3}, s=${node.params.strideRows || 1}×${node.params.strideCols || 1}, p=${node.params.padding || 0})`,
  controls: [
    { key: 'kernelRows', label: '창 높이', type: 'number', step: '1', default: 3 },
    { key: 'kernelCols', label: '창 너비', type: 'number', step: '1', default: 3 },
    { key: 'strideRows', label: '세로 stride', type: 'number', step: '1', default: 1 },
    { key: 'strideCols', label: '가로 stride', type: 'number', step: '1', default: 1 },
    { key: 'padding', label: 'zero padding', type: 'number', step: '1', default: 0 }
  ],
  compute: (node, [patches, like]) => fold2dValues(node, patches, like)
};

// The base HTML predates this complementary transform. Add it beside unfold2d.
const unfoldPaletteButton = document.querySelector('[data-block="unfold2d"]');
if (unfoldPaletteButton && !document.querySelector('[data-block="fold2d"]')) {
  const button = document.createElement('button');
  button.className = 'palette-block transform';
  button.dataset.block = 'fold2d';
  button.textContent = '슬라이딩 창 합치기';
  unfoldPaletteButton.insertAdjacentElement('afterend', button);
}

// Without a 미분 block there is no reliable way for the weight exporter to infer
// which state variables are model parameters and which are gradient accumulators.
// Make that educational distinction explicit on 변수 blocks instead.
const collectLegacyTrainableVariableNames = collectTrainableVariableNames;
collectTrainableVariableNames = function collectManualTrainableVariableNames() {
  const explicit = [];
  for (const node of graph.nodes.values()) {
    if (node.type !== 'variable') continue;
    if (String(node.params.trainable || '') !== 'yes') continue;
    const name = String(node.params.name || '').trim();
    if (name) explicit.push(name);
  }
  if (explicit.length) return [...new Set(explicit)];

  // Old, not-yet-converted projects may still contain legacy derivative nodes.
  // Read those raw node records directly; the manual runtime intentionally no
  // longer exposes derivative as an executable block type.
  const updateTargets = collectSetVariableTargets();
  const legacy = [];
  for (const node of graph.nodes.values()) {
    if (node.type !== 'derivative') continue;
    const name = String(node.params.variable || '').trim();
    if (name && updateTargets.has(name) && findVariableNode(name)) legacy.push(name);
  }
  if (legacy.length) return [...new Set(legacy)];

  // Keep the old helper as a last compatibility attempt, but it may return an
  // empty list in a fully manual graph.
  try { return collectLegacyTrainableVariableNames(); }
  catch { return []; }
};

INSPECTOR_EXTENSIONS.push(node => {
  if (inspectorContent.hidden || node.type !== 'variable') return;

  const row = document.createElement('div');
  row.className = 'control-row';
  const label = document.createElement('label');
  label.textContent = '가중치 저장 분류';
  const select = document.createElement('select');
  select.append(new Option('미지정', 'auto'));
  select.append(new Option('학습 파라미터', 'yes'));
  select.append(new Option('보조/누적 변수', 'no'));
  select.value = String(node.params.trainable || 'auto');
  select.addEventListener('input', () => {
    node.params.trainable = select.value;
    notifyWorkspaceChanged();
  });
  row.append(label, select);
  inspectorControls.appendChild(row);
});
