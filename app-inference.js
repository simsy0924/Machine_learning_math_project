// Generic inference helpers: argmax/equality blocks and classify the current drawing
// through a selected pure model-output node without changing the trained weights.

function argmaxValue(value) {
  const array = asArrayValue(value);
  if (array.shape.length !== 1) throw new Error('최댓값 위치에는 벡터가 필요합니다.');
  if (!array.data.length) throw new Error('빈 벡터에서는 최댓값 위치를 구할 수 없습니다.');
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

BLOCKS.argmax = {
  title: '최댓값 위치',
  kind: 'operation',
  inputs: ['벡터'],
  description: '벡터에서 가장 큰 원소의 위치를 0부터 시작하는 번호로 반환한다. 분류 결과를 종류 번호로 바꿀 때 사용할 수 있다.',
  formula: () => 'argmax(x)',
  compute: (node, [value]) => argmaxValue(value)
};

BLOCKS.equal = {
  title: '같음?',
  kind: 'operation',
  inputs: ['a', 'b'],
  description: '두 숫자가 같으면 1, 다르면 0을 반환한다. 예측 종류와 정답 종류를 비교해 정확도를 계산할 때 사용할 수 있다.',
  formula: () => '[a = b]',
  compute: (node, [a, b]) => {
    if (typeof a !== 'number' || typeof b !== 'number') throw new Error('같음? 블록에는 숫자 두 개가 필요합니다.');
    return a === b ? 1 : 0;
  }
};

const primitiveVJPBeforeInference = primitiveVJP;
primitiveVJP = function(type, inputs, output, upstream) {
  // argmax/equality are evaluation metrics, not useful training-loss operations.
  // Their derivative is treated as zero almost everywhere.
  if (type === 'argmax') return [zerosLike(inputs[0])];
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
      if (def.special === 'repeat' || def.special === 'setVariable') {
        throw new Error('학습/값 변경 블록이 포함된 출력입니다. 순수한 예측 점수 출력 블록을 선택하세요.');
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
