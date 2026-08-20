// Fast inference path for evaluating one selected pure output without walking the
// whole workspace. This is useful after training: test a new Quick Draw sample by
// changing test_i and evaluating only the prediction output, without repeating SGD.

const evaluateSelectedBtn = document.getElementById('evaluateSelectedBtn');

function assertPureSelectedOutput(nodeId) {
  const ancestors = collectAncestorNodeIds(nodeId);
  for (const id of ancestors) {
    const node = graph.nodes.get(id);
    if (!node) continue;
    const special = getBlockDef(node.type).special;
    if (special === 'repeat' || special === 'setVariable' || special === 'derivative') {
      throw new Error('선택한 출력에 반복·값 변경·미분이 포함되어 있습니다. 순수한 테스트/예측 출력 블록을 선택하세요.');
    }
  }
  return ancestors;
}

function selectedPredictionSummary(value) {
  if (!isArrayValue(value)) return null;
  const vector = asArrayValue(value);
  if (vector.shape.length !== 1) return null;
  const classes = window.quickDrawDataset?.getLoadedClassNames?.() || [];
  if (!classes.length || vector.data.length !== classes.length) return null;
  const index = argmaxValue(vector);
  const sum = Array.from(vector.data).reduce((a, b) => a + b, 0);
  const isProbability = Array.from(vector.data).every(v => v >= -1e-5 && v <= 1.00001) && Math.abs(sum - 1) < 0.02;
  const best = Number(vector.data[index]);
  const score = isProbability ? `${(best * 100).toFixed(1)}%` : best.toFixed(4);
  return `예측 ${koreanClassName(classes[index])} · ${score}`;
}

function evaluateSelectedOutput() {
  try {
    if (selectedNodeId == null) throw new Error('먼저 테스트할 출력 블록을 선택하세요.');
    assertPureSelectedOutput(selectedNodeId);
    const value = evaluateNode(selectedNodeId, new Map(), new Set());
    const node = graph.nodes.get(selectedNodeId);
    if (node) {
      node.lastValue = copyValue(value);
      node.lastError = null;
      updateNodePreview(node);
    }
    renderInspector();
    const summary = selectedPredictionSummary(value);
    workspaceStatus.textContent = summary ? `선택 계산 완료 · ${summary}` : '선택 계산 완료';
  } catch (error) {
    console.error(error);
    workspaceStatus.textContent = `선택 계산 오류 · ${error.message}`;
    alert(`선택 계산 실패: ${error.message}`);
  }
}

evaluateSelectedBtn?.addEventListener('click', evaluateSelectedOutput);
