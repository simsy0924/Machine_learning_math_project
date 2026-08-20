// Selected-branch execution and general-purpose bulk evaluation helpers.
// The goal is to keep evaluation expressed with ordinary math blocks: repeat
// ranges, outer products, sums, equality tests, and runtime variables.

const evaluateSelectedBtn = document.getElementById('evaluateSelectedBtn');

// ---------- General math additions ----------

function repeatStartValue(node) {
  return Math.floor(Number(node?.params?.start) || 0);
}

if (!BLOCKS.repeat.controls.some(control => control.key === 'start')) {
  BLOCKS.repeat.controls.splice(1, 0, {
    key: 'start',
    label: '반복 시작값',
    type: 'number',
    step: '1',
    default: 0
  });
}
BLOCKS.repeat.description = '연결된 계산을 여러 번 다시 실행한다. 반복 번호는 시작값부터 1씩 증가한다.';
BLOCKS.repeat.formula = node => {
  const start = repeatStartValue(node);
  const count = repeatCount(node);
  const end = count > 0 ? start + count - 1 : start;
  return `repeat ${count} times (${node.params.indexVariable || 'i'}=${start}…${end})`;
};

// Equality follows the same scalar/array broadcasting rule as the ordinary
// arithmetic blocks. This lets max(p) be compared with every element of p
// without adding an AI-specific argmax block.
BLOCKS.equal.description = '숫자나 배열을 원소별로 비교한다. 같으면 1, 다르면 0을 반환한다.';
BLOCKS.equal.compute = (node, [a, b]) => elementwiseBinary(a, b, (x, y) => x === y ? 1 : 0);

BLOCKS.outerProduct = {
  title: '외적',
  kind: 'operation',
  inputs: ['a', 'b'],
  description: '두 벡터의 외적 a⊗b를 계산해 행렬을 만든다. 종류별 확률 누적 같은 계산에도 사용할 수 있다.',
  formula: () => 'a ⊗ b',
  compute: (node, [a, b]) => outerProductValues(a, b)
};

(function installOuterProductPaletteButton() {
  if (document.querySelector('[data-block="outerProduct"]')) return;
  const dotButton = document.querySelector('[data-block="dot"]');
  const operationGrid = dotButton?.parentElement;
  if (!operationGrid) return;
  const button = document.createElement('button');
  button.className = 'palette-block operation';
  button.dataset.block = 'outerProduct';
  button.textContent = '외적';
  operationGrid.appendChild(button);
})();

const primitiveVJPBeforeBulkEvaluation = primitiveVJP;
primitiveVJP = function(type, inputs, output, upstream) {
  if (type === 'outerProduct') {
    const [a, b] = inputs;
    const aa = asArrayValue(a);
    const bb = asArrayValue(b);
    const matrix = asArrayValue(upstream);
    if (matrix.shape.length !== 2 || matrix.shape[0] !== aa.data.length || matrix.shape[1] !== bb.data.length) {
      throw new Error('외적의 미분 행렬 크기가 맞지 않습니다.');
    }
    return [
      matrixVectorValues(matrix, bb),
      transposeMatrixVectorValues(matrix, aa)
    ];
  }
  return primitiveVJPBeforeBulkEvaluation(type, inputs, output, upstream);
};

// ---------- Repeat ranges ----------
// app-training-runtime already supplies the safe nested-loop execution context.
// These overrides keep those semantics and only change the loop value from
// 0..count-1 to start..start+count-1.

const evaluateNodeBeforeRepeatRange = evaluateNode;
evaluateNode = function(nodeId, memo = new Map(), visiting = new Set()) {
  if (memo.has(nodeId)) return memo.get(nodeId);
  const node = graph.nodes.get(nodeId);
  if (!node) throw new Error('존재하지 않는 블록입니다.');
  const def = getBlockDef(node.type);

  if (def.special !== 'repeat') {
    return evaluateNodeBeforeRepeatRange(nodeId, memo, visiting);
  }

  if (visiting.has(nodeId)) throw new Error('순환 연결은 계산할 수 없습니다.');
  visiting.add(nodeId);

  const structureCache = ensureGraphStructureCache();
  const connection = cachedGraphInput(structureCache, nodeId, 0);
  if (!connection) {
    visiting.delete(nodeId);
    throw new Error("입력 '실행할 식'이 연결되지 않았습니다.");
  }

  const count = repeatCount(node);
  const start = repeatStartValue(node);
  const indexName = String(node.params.indexVariable || 'i');
  let last = 0;
  let lastIndex = null;

  try {
    for (let offset = 0; offset < count; offset++) {
      const loopValue = start + offset;
      lastIndex = loopValue;
      const frame = new Map([[indexName, loopValue]]);
      const parentPendingUpdates = pendingVariableUpdates;

      LOOP_CONTEXT_STACK.push(frame);
      pendingVariableUpdates = new Map();
      try {
        last = evaluateNode(connection.from, new Map(), new Set());
        commitPendingVariableUpdates();
      } finally {
        pendingVariableUpdates = parentPendingUpdates;
        LOOP_CONTEXT_STACK.pop();
      }
    }

    if (lastIndex != null && findVariableNode(indexName)) {
      writeRuntimeVariable(indexName, lastIndex, true);
    }

    memo.set(nodeId, last);
    return last;
  } finally {
    visiting.delete(nodeId);
  }
};

const evaluateRepeatProgressiveBeforeRange = evaluateRepeatProgressive;
evaluateRepeatProgressive = async function(nodeId, memo, visiting, progress) {
  if (memo.has(nodeId)) return memo.get(nodeId);
  if (visiting.has(nodeId)) throw new Error('순환 연결은 계산할 수 없습니다.');

  const node = graph.nodes.get(nodeId);
  if (!node) throw new Error('존재하지 않는 블록입니다.');
  const structureCache = progress.graphCache || ensureGraphStructureCache();
  const connection = cachedGraphInput(structureCache, nodeId, 0);
  if (!connection) throw new Error("입력 '실행할 식'이 연결되지 않았습니다.");

  visiting.add(nodeId);
  const count = repeatCount(node);
  const start = repeatStartValue(node);
  const indexName = String(node.params.indexVariable || 'i');
  const isLeafRepeat = !branchContainsRepeat(connection.from, new Set(), structureCache);
  let last = 0;
  let lastIndex = null;

  try {
    for (let offset = 0; offset < count; offset++) {
      const loopValue = start + offset;
      lastIndex = loopValue;
      const loopFrame = new Map([[indexName, loopValue]]);
      const parentPendingUpdates = pendingVariableUpdates;
      const progressFrame = {
        name: `${indexName}=${loopValue}`,
        index: offset,
        count
      };

      LOOP_CONTEXT_STACK.push(loopFrame);
      progress.stack.push(progressFrame);
      pendingVariableUpdates = new Map();
      try {
        last = await evaluateNodeProgressive(connection.from, new Map(), new Set(), progress);
        commitPendingVariableUpdates();
        if (isLeafRepeat) {
          progress.completed++;
          await maybeYieldProgress(progress);
        }
      } finally {
        pendingVariableUpdates = parentPendingUpdates;
        progress.stack.pop();
        LOOP_CONTEXT_STACK.pop();
      }
    }

    if (lastIndex != null && findVariableNode(indexName)) {
      writeRuntimeVariable(indexName, lastIndex, true);
    }

    memo.set(nodeId, last);
    return last;
  } finally {
    visiting.delete(nodeId);
  }
};

// ---------- Selected result UI ----------

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

function koreanEvaluationClassName(name) {
  if (typeof koreanClassName === 'function') return koreanClassName(name);
  return name;
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

function renderSelectedEvaluationResult(value) {
  const target = selectedResultElement();
  if (!target) return;
  target.replaceChildren();

  if (typeof value === 'number') {
    target.textContent = formatSelectedScalar(value);
    return;
  }

  if (!isArrayValue(value)) {
    target.textContent = '계산은 완료됐지만 이 값은 숫자/배열 형식이 아닙니다.';
    return;
  }

  const array = asArrayValue(value);
  const classes = window.quickDrawDataset?.getLoadedClassNames?.() || [];

  if (array.shape.length === 1 && classes.length === array.data.length) {
    const probability = isProbabilityVector(array);
    const bestIndex = argmaxValue(array);
    const heading = document.createElement('div');
    heading.style.marginBottom = '8px';
    heading.innerHTML = `<strong>예측: ${escapeHtml(koreanEvaluationClassName(classes[bestIndex]))}</strong>${probability ? ` · ${(array.data[bestIndex] * 100).toFixed(1)}%` : ''}`;
    target.appendChild(heading);

    const table = document.createElement('table');
    table.style.width = '100%';
    table.style.borderCollapse = 'collapse';
    const body = document.createElement('tbody');
    for (let i = 0; i < classes.length; i++) {
      const row = document.createElement('tr');
      const label = document.createElement('td');
      const score = document.createElement('td');
      label.textContent = koreanEvaluationClassName(classes[i]);
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
    return;
  }

  if (array.shape.length === 2) {
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
        th.textContent = koreanEvaluationClassName(className);
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
        th.textContent = koreanEvaluationClassName(classes[r]);
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
    return;
  }

  const previewCount = Math.min(40, array.data.length);
  const preview = Array.from(array.data.slice(0, previewCount), value => Number(value).toFixed(5)).join(', ');
  target.textContent = `shape [${array.shape.join('×')}] · [${preview}${array.data.length > previewCount ? ', …' : ''}]`;
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
  return `예측 ${koreanEvaluationClassName(classes[index])} · ${score}`;
}

function selectedBranchSpecials(nodeId) {
  const ancestors = collectAncestorNodeIds(nodeId);
  const specials = new Set();
  for (const id of ancestors) {
    const node = graph.nodes.get(id);
    if (!node) continue;
    const special = getBlockDef(node.type).special;
    if (special) specials.add(special);
  }
  return { ancestors, specials };
}

async function evaluateSelectedStatefulBranch(nodeId) {
  let graphCache = null;
  try {
    graphCache = beginGraphStructureRun();
    const progress = {
      total: repeatLeafWork(nodeId, new Set(), graphCache),
      completed: 0,
      stack: [],
      lastYield: performance.now(),
      graphCache
    };

    if (progress.total) {
      workspaceStatus.textContent = formatProgressStatus(progress);
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    const value = await evaluateNodeProgressive(nodeId, new Map(), new Set(), progress);
    if (progress.total) await maybeYieldProgress(progress, true);
    return { value, progress };
  } finally {
    if (graphCache) endGraphStructureRun();
  }
}

async function evaluateSelectedOutput() {
  if (evaluateSelectedBtn?.disabled) return;
  const oldText = evaluateSelectedBtn?.textContent || '선택 계산';

  try {
    if (selectedNodeId == null) throw new Error('먼저 계산할 블록을 선택하세요.');
    const { specials } = selectedBranchSpecials(selectedNodeId);
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
