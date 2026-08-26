// 선택 계산: run just the selected block's branch and show the result in detail.
// Useful for testing a trained model on many samples without evaluating the
// whole workspace.

function ensureSelectedEvaluationPanel() {
  let panel = document.getElementById('selectedEvaluationPanel');
  if (panel) return panel;

  const inspector = document.querySelector('.inspector-panel');
  if (!inspector) return null;

  const section = document.createElement('section');
  section.className = 'inspector-card';
  section.id = 'selectedEvaluationPanel';
  section.innerHTML = `
    <div class="section-head"><h2>선택 계산 결과</h2></div>
    <div id="selectedEvaluationResult" class="dataset-status">선택 계산을 실행하면 숫자, 확률 벡터, 행렬 결과를 자세히 표시합니다.</div>`;
  inspector.insertBefore(section, inspector.children[1] || null);
  return section;
}

ensureSelectedEvaluationPanel();

function selectedResultElement() {
  return document.getElementById('selectedEvaluationResult');
}

function isProbabilityVector(vector) {
  if (vector.shape.length !== 1 || !vector.data.length) return false;
  let sum = 0;
  for (const value of vector.data) {
    if (!Number.isFinite(value) || value < -1e-5 || value > 1.00001) return false;
    sum += value;
  }
  return Math.abs(sum - 1) < 0.02;
}

function isProbabilityMatrix(matrix) {
  if (matrix.shape.length !== 2) return false;
  const [rows, cols] = matrix.shape;
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    for (let c = 0; c < cols; c++) {
      const value = matrix.data[r * cols + c];
      if (!Number.isFinite(value) || value < -1e-5 || value > 1.00001) return false;
      sum += value;
    }
    if (Math.abs(sum - 1) >= 0.03) return false;
  }
  return true;
}

function formatSelectedScalar(value) {
  if (!Number.isFinite(value)) return String(value);
  return Number.isInteger(value) ? value.toLocaleString('ko-KR') : value.toFixed(6);
}

function renderSelectedScoreTable(target, array, classes) {
  const probability = isProbabilityVector(array);
  const bestIndex = argmaxValue(array);
  const heading = document.createElement('div');
  heading.style.marginBottom = '8px';
  heading.innerHTML = `<strong>예측: ${escapeHtml(koreanClassName(classes[bestIndex]))}</strong>${probability ? ` · ${(array.data[bestIndex] * 100).toFixed(1)}%` : ''}`;
  target.appendChild(heading);

  const table = document.createElement('table');
  table.style.width = '100%';
  table.style.borderCollapse = 'collapse';
  const body = document.createElement('tbody');
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('tr');
    const label = document.createElement('td');
    const score = document.createElement('td');
    label.textContent = koreanClassName(classes[i]);
    score.textContent = probability ? `${(array.data[i] * 100).toFixed(2)}%` : Number(array.data[i]).toFixed(6);
    score.style.textAlign = 'right';
    if (i === bestIndex) {
      label.style.fontWeight = '700';
      score.style.fontWeight = '700';
    }
    row.append(label, score);
    body.appendChild(row);
  }
  table.appendChild(body);
  target.appendChild(table);
}

function renderSelectedMatrix(target, array, classes) {
  const [rows, cols] = array.shape;
  const probability = isProbabilityMatrix(array);
  const useClasses = classes.length === rows && classes.length === cols;
  const caption = document.createElement('div');
  caption.style.marginBottom = '8px';
  caption.textContent = `${rows}×${cols} 행렬${probability ? ' · 각 행을 확률로 표시' : ''}`;
  target.appendChild(caption);

  const wrapper = document.createElement('div');
  wrapper.style.overflowX = 'auto';
  const table = document.createElement('table');
  table.style.borderCollapse = 'collapse';
  table.style.minWidth = '100%';

  if (useClasses) {
    const head = document.createElement('thead');
    const row = document.createElement('tr');
    const corner = document.createElement('th');
    corner.textContent = '실제 \\ 출력';
    row.appendChild(corner);
    for (const className of classes) {
      const th = document.createElement('th');
      th.textContent = koreanClassName(className);
      th.style.padding = '3px 6px';
      row.appendChild(th);
    }
    head.appendChild(row);
    table.appendChild(head);
  }

  const body = document.createElement('tbody');
  for (let r = 0; r < rows; r++) {
    const row = document.createElement('tr');
    if (useClasses) {
      const th = document.createElement('th');
      th.textContent = koreanClassName(classes[r]);
      th.style.textAlign = 'left';
      th.style.padding = '3px 6px';
      row.appendChild(th);
    }
    for (let c = 0; c < cols; c++) {
      const td = document.createElement('td');
      const number = array.data[r * cols + c];
      td.textContent = probability ? `${(number * 100).toFixed(1)}%` : Number(number).toFixed(4);
      td.style.textAlign = 'right';
      td.style.padding = '3px 6px';
      row.appendChild(td);
    }
    body.appendChild(row);
  }
  table.appendChild(body);
  wrapper.appendChild(table);
  target.appendChild(wrapper);
}

function renderSelectedEvaluationResult(value) {
  const target = selectedResultElement();
  if (!target) return;
  target.replaceChildren();

  if (typeof value === 'number') {
    target.textContent = formatSelectedScalar(value);
  } else if (!isArrayValue(value)) {
    target.textContent = '계산은 완료됐지만 이 값은 숫자/배열 형식이 아닙니다.';
  } else {
    const array = asArrayValue(value);
    const classes = window.quickDrawDataset?.getLoadedClassNames?.() || [];

    if (array.shape.length === 1 && classes.length === array.data.length) {
      renderSelectedScoreTable(target, array, classes);
    } else if (array.shape.length === 2) {
      renderSelectedMatrix(target, array, classes);
    } else {
      const previewCount = Math.min(40, array.data.length);
      const preview = Array.from(array.data.slice(0, previewCount), v => Number(v).toFixed(5)).join(', ');
      target.textContent = `shape [${array.shape.join('×')}] · [${preview}${array.data.length > previewCount ? ', …' : ''}]`;
    }
  }

  for (const extend of SELECTED_RESULT_EXTENSIONS) extend(value);
}

function selectedPredictionSummary(value) {
  if (!isArrayValue(value)) return null;
  const vector = asArrayValue(value);
  if (vector.shape.length !== 1) return null;
  const classes = window.quickDrawDataset?.getLoadedClassNames?.() || [];
  if (!classes.length || vector.data.length !== classes.length) return null;
  const index = argmaxValue(vector);
  const probability = isProbabilityVector(vector);
  const best = Number(vector.data[index]);
  const score = probability ? `${(best * 100).toFixed(1)}%` : best.toFixed(4);
  return `예측 ${koreanClassName(classes[index])} · ${score}`;
}

function selectedBranchSpecials(nodeId) {
  const specials = new Set();
  for (const id of collectAncestorNodeIds(nodeId)) {
    const node = graph.nodes.get(id);
    if (!node) continue;
    const special = getBlockDef(node.type).special;
    if (special) specials.add(special);
  }
  return specials;
}

async function evaluateSelectedOutput() {
  if (evaluateSelectedBtn?.disabled) return;
  const oldText = evaluateSelectedBtn?.textContent || '선택 계산';

  try {
    if (selectedNodeId == null) throw new Error('먼저 계산할 블록을 선택하세요.');
    const specials = selectedBranchSpecials(selectedNodeId);
    if (specials.has('derivative') && (specials.has('repeat') || specials.has('setVariable'))) {
      throw new Error('선택 실행에서는 반복/값 변경과 미분을 한 가지에 섞지 마세요. 테스트 가지에는 미분 블록이 필요하지 않습니다.');
    }

    if (evaluateSelectedBtn) {
      evaluateSelectedBtn.disabled = true;
      evaluateSelectedBtn.textContent = '선택 계산 중…';
    }

    let value;
    let progress = null;
    if (specials.has('repeat') || specials.has('setVariable')) {
      const result = await evaluateSelectedStatefulBranch(selectedNodeId);
      value = result.value;
      progress = result.progress;
    } else {
      value = evaluateNode(selectedNodeId, new Map(), new Set());
    }

    const node = graph.nodes.get(selectedNodeId);
    if (node) {
      node.lastValue = copyValue(value);
      node.lastError = null;
      updateNodePreview(node);
    }
    renderInspector();
    renderSelectedEvaluationResult(value);

    const summary = selectedPredictionSummary(value);
    if (progress?.total) {
      const progressText = formatProgressStatus(progress);
      workspaceStatus.textContent = summary
        ? `선택 계산 완료 · ${progressText} · ${summary}`
        : `선택 계산 완료 · ${progressText}`;
    } else {
      workspaceStatus.textContent = summary ? `선택 계산 완료 · ${summary}` : '선택 계산 완료';
    }
  } catch (error) {
    console.error(error);
    workspaceStatus.textContent = `선택 계산 오류 · ${error.message}`;
    const target = selectedResultElement();
    if (target) target.textContent = `오류: ${error.message}`;
    alert(`선택 계산 실패: ${error.message}`);
  } finally {
    if (evaluateSelectedBtn) {
      evaluateSelectedBtn.disabled = false;
      evaluateSelectedBtn.textContent = oldText;
    }
  }
}

evaluateSelectedBtn?.addEventListener('click', evaluateSelectedOutput);
