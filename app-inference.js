// Generic evaluation helpers and drawing classification tools.
// The UI may internally choose the largest score, but argmax is intentionally
// not exposed as a math block. Higher-level AI operations should be assembled
// from ordinary mathematical blocks whenever possible.

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

BLOCKS.arrayMax = {
  title: '배열 최댓값',
  kind: 'operation',
  inputs: ['배열'],
  description: '배열의 모든 원소 중 가장 큰 값 하나를 반환한다. 안정화된 지수 계산이나 Softmax 같은 수식을 직접 만들 때 사용할 수 있다.',
  formula: () => 'maxᵢ xᵢ',
  compute: (node, [value]) => arrayMaximumValue(value)
};

BLOCKS.equal = {
  title: '같음?',
  kind: 'operation',
  inputs: ['a', 'b'],
  description: '두 숫자가 같으면 1, 다르면 0을 반환한다.',
  formula: () => '[a = b]',
  compute: (node, [a, b]) => {
    if (typeof a !== 'number' || typeof b !== 'number') throw new Error('같음? 블록에는 숫자 두 개가 필요합니다.');
    return a === b ? 1 : 0;
  }
};

const primitiveVJPBeforeInference = primitiveVJP;
primitiveVJP = function(type, inputs, output, upstream) {
  if (type === 'arrayMax') {
    const input = asArrayValue(inputs[0]);
    const maximum = Number(output);
    let tieCount = 0;
    for (const value of input.data) if (value === maximum) tieCount++;
    if (!tieCount) return [zerosLike(input)];

    const upstreamScalar = typeof upstream === 'number' ? upstream : sumArray(asArrayValue(upstream));
    const share = upstreamScalar / tieCount;
    const grad = new Float32Array(input.data.length);
    for (let i = 0; i < input.data.length; i++) {
      if (input.data[i] === maximum) grad[i] = share;
    }
    return [arrayValue(grad, input.shape)];
  }
  // Equality is a discrete comparison; use zero derivative almost everywhere.
  if (type === 'equal') return [zerosLike(inputs[0]), zerosLike(inputs[1])];
  return primitiveVJPBeforeInference(type, inputs, output, upstream);
};

function collectAncestorNodeIds(outputId) {
  const visited = new Set();
  const visit = id => {
    if (visited.has(id)) return;
    visited.add(id);
    const node = graph.nodes.get(id);
    if (!node) return;
    const def = getBlockDef(node.type);
    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const connection = graph.connections.find(c => c.to === id && c.inputIndex === inputIndex);
      if (connection) visit(connection.from);
    }
  };
  visit(outputId);
  return visited;
}

function koreanClassName(name) {
  const found = DATA_CLASS_OPTIONS.find(([value]) => value === name);
  return found ? found[1] : name;
}

function formatClassifierScores(classes, scores) {
  return classes.map((name, i) => `${koreanClassName(name)} ${Number(scores.data[i]).toFixed(3)}`).join(' · ');
}

function setDrawingClassifierResult(message, isError = false) {
  const result = document.getElementById('drawingClassificationResult');
  if (!result) return;
  result.textContent = message;
  result.classList.toggle('classification-error', isError);
}

function classifyCurrentDrawing() {
  try {
    if (selectedNodeId == null) throw new Error('먼저 작업공간에서 모델의 최종 점수 출력 블록을 선택하세요.');

    const classes = window.quickDrawDataset?.getLoadedClassNames?.() || [];
    if (!classes.length) throw new Error('먼저 Quick Draw 종류를 불러오세요.');

    const ancestors = collectAncestorNodeIds(selectedNodeId);
    let hasSampleImage = false;
    for (const id of ancestors) {
      const node = graph.nodes.get(id);
      if (!node) continue;
      const def = getBlockDef(node.type);
      if (node.type === 'sampleImage') hasSampleImage = true;
      if (def.special === 'repeat' || def.special === 'setVariable' || def.special === 'derivative') {
        throw new Error('학습·값 변경·미분 블록이 포함된 출력입니다. 순수한 예측 점수 출력 블록을 선택하세요.');
      }
    }
    if (!hasSampleImage) {
      throw new Error("선택한 출력이 '그림값' 블록을 사용하지 않습니다. 학습된 모델의 점수 출력 블록을 선택하세요.");
    }

    const drawing = preprocessCanvasTo28(drawCanvas);
    const inkAmount = sumArray(drawing);
    if (inkAmount <= 0.001) throw new Error('그림판에 먼저 그림을 그리세요.');
    renderPreview(drawing.data);

    const originalSampleImageCompute = BLOCKS.sampleImage.compute;
    BLOCKS.sampleImage.compute = () => arrayValue(new Float32Array(drawing.data), [28, 28]);

    let scores;
    try {
      scores = evaluateNode(selectedNodeId, new Map(), new Set());
    } finally {
      BLOCKS.sampleImage.compute = originalSampleImageCompute;
    }

    const vector = asArrayValue(scores);
    if (vector.shape.length !== 1) throw new Error('선택한 출력이 점수 벡터가 아닙니다.');
    if (vector.data.length !== classes.length) {
      throw new Error(`출력 길이 ${vector.data.length}와 현재 데이터 종류 수 ${classes.length}가 다릅니다.`);
    }

    const index = argmaxValue(vector);
    const className = classes[index];
    const label = koreanClassName(className);
    setDrawingClassifierResult(`${label} (${className}) · ${formatClassifierScores(classes, vector)}`);

    const selected = graph.nodes.get(selectedNodeId);
    if (selected) {
      selected.lastValue = copyValue(vector);
      selected.lastError = null;
      updateNodePreview(selected);
      renderInspector();
    }
  } catch (error) {
    setDrawingClassifierResult(error.message, true);
  }
}

document.getElementById('classifyDrawingBtn')?.addEventListener('click', classifyCurrentDrawing);
