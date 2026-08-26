// Export and import only variables that are actual trainable model parameters.
//
// A trainable parameter is identified by the variable named on a derivative
// block, and it must also be the target of at least one setVariable update.
// This works for both direct SGD and minibatch training. In minibatch graphs the
// derivative flows into accumulator variables such as gW, while the actual model
// parameter W is updated later from that accumulator; following the derivative
// branch into setVariable would therefore save the wrong variable.

const WEIGHTS_FORMAT = 'machine-learning-math-weights';
const WEIGHTS_VERSION = 1;
const MAX_WEIGHTS_FILE_BYTES = 50 * 1024 * 1024;

const exportWeightsBtn = document.getElementById('exportWeightsBtn');
const importWeightsBtn = document.getElementById('importWeightsBtn');
const importWeightsInput = document.getElementById('importWeightsInput');

function collectSetVariableTargets() {
  const names = new Set();
  for (const node of graph.nodes.values()) {
    const def = getBlockDef(node.type);
    if (def.special !== 'setVariable') continue;
    const name = String(node.params.variable || '').trim();
    if (name) names.add(name);
  }
  return names;
}

function collectTrainableVariableNames() {
  const updateTargets = collectSetVariableTargets();
  const names = new Set();

  for (const node of graph.nodes.values()) {
    const def = getBlockDef(node.type);
    if (def.special !== 'derivative') continue;

    const name = String(node.params.variable || '').trim();
    if (!name || !updateTargets.has(name)) continue;
    if (!findVariableNode(name)) continue;
    names.add(name);
  }

  return [...names];
}

function buildWeightsSnapshot() {
  const names = collectTrainableVariableNames();
  if (!names.length) throw new Error("미분되고 실제로 '값 바꾸기'로 갱신되는 학습 변수가 없습니다.");

  const variables = names.map(name => {
    const variableNode = findVariableNode(name);
    if (!variableNode) throw new Error(`'${name}' 변수 블록을 찾지 못했습니다.`);
    const value = readRuntimeVariable(variableNode);
    return {
      name,
      value: serializeRuntimeValue(value)
    };
  });

  return {
    format: WEIGHTS_FORMAT,
    version: WEIGHTS_VERSION,
    exportedAt: new Date().toISOString(),
    classes: window.quickDrawDataset?.getLoadedClassNames?.() || [],
    variables
  };
}

function makeDownload(text, filename) {
  const blob = new Blob([text], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportWeights() {
  if (evaluateBtn?.disabled) {
    alert('계산이 끝난 뒤 가중치를 저장하세요.');
    return;
  }
  try {
    const snapshot = buildWeightsSnapshot();
    const now = new Date();
    const stamp = [
      now.getFullYear(),
      String(now.getMonth() + 1).padStart(2, '0'),
      String(now.getDate()).padStart(2, '0'),
      '-',
      String(now.getHours()).padStart(2, '0'),
      String(now.getMinutes()).padStart(2, '0')
    ].join('');
    makeDownload(JSON.stringify(snapshot), `math-weights-${stamp}.mmlweights`);
    workspaceStatus.textContent = `가중치 ${snapshot.variables.length}개 저장 완료`;
  } catch (error) {
    console.error(error);
    alert(`가중치 저장 실패: ${error.message}`);
  }
}

function validateWeightsSnapshot(snapshot) {
  if (!snapshot || snapshot.format !== WEIGHTS_FORMAT) throw new Error('이 프로젝트의 가중치 파일이 아닙니다.');
  if (snapshot.version !== WEIGHTS_VERSION) throw new Error(`지원하지 않는 가중치 버전입니다: ${snapshot.version}`);
  if (!Array.isArray(snapshot.variables) || !snapshot.variables.length) throw new Error('저장된 가중치가 없습니다.');
  if (snapshot.variables.length > 1000) throw new Error('저장된 변수 수가 지나치게 많습니다.');
}

function assertTrainableVariableSet(snapshot) {
  const expected = collectTrainableVariableNames();
  if (!expected.length) throw new Error('현재 그래프에서 학습 가중치를 찾지 못했습니다.');

  const saved = snapshot.variables.map(item => String(item?.name || '').trim());
  if (saved.some(name => !name)) throw new Error('이름이 없는 가중치가 있습니다.');
  if (new Set(saved).size !== saved.length) throw new Error('가중치 파일에 같은 변수 이름이 중복되어 있습니다.');

  const expectedSet = new Set(expected);
  const savedSet = new Set(saved);
  const missing = expected.filter(name => !savedSet.has(name));
  const unexpected = saved.filter(name => !expectedSet.has(name));

  if (missing.length || unexpected.length) {
    const details = [];
    if (missing.length) details.push(`빠진 학습 가중치: ${missing.join(', ')}`);
    if (unexpected.length) details.push(`현재 모델의 학습 가중치가 아닌 변수: ${unexpected.join(', ')}`);
    throw new Error(`가중치 파일이 현재 모델과 맞지 않습니다. ${details.join(' / ')}`);
  }
}

function sameArrayShape(a, b) {
  return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => Number(v) === Number(b[i]));
}

function assertCompatibleVariable(name, variableNode, value) {
  const expected = initialVariableValue(variableNode);
  if (typeof expected === 'number') {
    if (typeof value !== 'number') throw new Error(`'${name}' 변수는 숫자여야 합니다.`);
    return;
  }
  if (!isArrayValue(expected) || !isArrayValue(value)) throw new Error(`'${name}' 변수 형식이 현재 모델과 다릅니다.`);
  if (!sameArrayShape(expected.shape, value.shape)) {
    throw new Error(`'${name}' 모양이 다릅니다. 현재 ${expected.shape.join('×')}, 파일 ${value.shape.join('×')}`);
  }
}

function assertClassOrderCompatible(savedClasses) {
  if (!Array.isArray(savedClasses) || !savedClasses.length) return;
  const current = window.quickDrawDataset?.getLoadedClassNames?.() || [];
  if (!current.length) return;
  if (current.length !== savedClasses.length || current.some((name, i) => name !== savedClasses[i])) {
    throw new Error(`데이터 종류 순서가 가중치 파일과 다릅니다. 파일: ${savedClasses.join(', ')} / 현재: ${current.join(', ')}`);
  }
}

function restoreWeights(snapshot) {
  validateWeightsSnapshot(snapshot);
  assertClassOrderCompatible(snapshot.classes);
  assertTrainableVariableSet(snapshot);

  const restored = [];
  for (const saved of snapshot.variables) {
    const name = String(saved?.name || '').trim();
    const variableNode = findVariableNode(name);
    if (!variableNode) throw new Error(`현재 모델에 '${name}' 변수 블록이 없습니다.`);
    const value = deserializeRuntimeValue(saved.value);
    assertCompatibleVariable(name, variableNode, value);
    RUNTIME_VARIABLES.set(name, {
      signature: variableSignature(variableNode),
      value: copyValue(value)
    });
    restored.push(name);
  }

  invalidatePreviews();
  renderInspector();
  workspaceStatus.textContent = `가중치 ${restored.length}개 불러옴 · ${restored.join(', ')}`;
  return restored;
}

async function importWeightsFile(file) {
  if (!file) return;
  if (file.size > MAX_WEIGHTS_FILE_BYTES) throw new Error('가중치 파일이 50MB보다 큽니다.');
  let snapshot;
  try { snapshot = JSON.parse(await file.text()); }
  catch { throw new Error('파일이 올바른 JSON 형식이 아닙니다.'); }
  return restoreWeights(snapshot);
}

exportWeightsBtn?.addEventListener('click', exportWeights);
importWeightsBtn?.addEventListener('click', () => importWeightsInput?.click());
importWeightsInput?.addEventListener('change', async () => {
  const file = importWeightsInput.files?.[0];
  if (!file) return;
  importWeightsBtn.disabled = true;
  exportWeightsBtn.disabled = true;
  try {
    const restored = await importWeightsFile(file);
    alert(`가중치 불러오기 완료: ${restored.join(', ')}`);
  } catch (error) {
    console.error(error);
    alert(`가중치 불러오기 실패: ${error.message}`);
  } finally {
    importWeightsInput.value = '';
    importWeightsBtn.disabled = false;
    exportWeightsBtn.disabled = false;
  }
});