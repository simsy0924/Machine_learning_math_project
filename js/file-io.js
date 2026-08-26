// Saving and loading files.
//
// Two formats, sharing one value serializer:
//   .mmlab       — a whole workspace: blocks, connections, 사용자 블록, runtime
//                  variable values, dataset selection and camera position.
//                  The Quick Draw images themselves are never embedded.
//   .mmlweights  — only the trained parameters of the current model.

const PROJECT_FORMAT = 'machine-learning-math-project';
const PROJECT_VERSION = 1;
const MAX_PROJECT_FILE_BYTES = 50 * 1024 * 1024;

const WEIGHTS_FORMAT = 'machine-learning-math-weights';
const WEIGHTS_VERSION = 1;
const MAX_WEIGHTS_FILE_BYTES = 50 * 1024 * 1024;

const exportProjectBtn = document.getElementById('exportProjectBtn');
const importProjectBtn = document.getElementById('importProjectBtn');
const importProjectInput = document.getElementById('importProjectInput');
const exportWeightsBtn = document.getElementById('exportWeightsBtn');
const importWeightsBtn = document.getElementById('importWeightsBtn');
const importWeightsInput = document.getElementById('importWeightsInput');

// ---------- shared helpers ----------

function bytesToBase64(bytes) {
  const chunkSize = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
    for (let j = 0; j < chunk.length; j++) binary += String.fromCharCode(chunk[j]);
  }
  return btoa(binary);
}

function base64ToBytes(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function serializeRuntimeValue(value) {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error('NaN 또는 무한대 값은 저장할 수 없습니다.');
    return { type: 'number', value };
  }
  if (isArrayValue(value)) {
    const array = asArrayValue(value);
    const bytes = new Uint8Array(array.data.buffer, array.data.byteOffset, array.data.byteLength);
    return { type: 'float32', shape: [...array.shape], data: bytesToBase64(bytes) };
  }
  throw new Error('저장할 수 없는 변수 값이 있습니다.');
}

function deserializeRuntimeValue(saved) {
  if (saved?.type === 'number') return Number(saved.value);
  if (saved?.type === 'float32') {
    if (!Array.isArray(saved.shape) || !saved.shape.length) throw new Error('배열 모양 정보가 없습니다.');
    const bytes = base64ToBytes(String(saved.data || ''));
    if (bytes.byteLength % 4 !== 0) throw new Error('Float32 데이터 크기가 올바르지 않습니다.');
    const data = new Float32Array(bytes.slice().buffer);
    const expectedLength = saved.shape.reduce((a, b) => a * Number(b), 1);
    if (data.length !== expectedLength) throw new Error('배열 모양과 저장된 데이터 길이가 다릅니다.');
    return arrayValue(data, saved.shape.map(Number));
  }
  throw new Error('알 수 없는 변수 저장 형식입니다.');
}

function serializeNode(node) {
  return {
    id: node.id,
    type: node.type,
    x: node.x,
    y: node.y,
    params: JSON.parse(JSON.stringify(node.params || {}))
  };
}

function downloadJson(text, filename) {
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

function fileStamp(now = new Date()) {
  return [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0'),
    '-',
    String(now.getHours()).padStart(2, '0'),
    String(now.getMinutes()).padStart(2, '0')
  ].join('');
}

// ---------- .mmlab: the whole workspace ----------

// `includeRuntimeVariables` is false for autosave, which deliberately never
// writes trained weights or gradient accumulators to localStorage.
function buildProjectSnapshot({ includeRuntimeVariables = true } = {}) {
  const runtimeVariables = [];
  if (includeRuntimeVariables) {
    for (const [name, entry] of RUNTIME_VARIABLES.entries()) {
      runtimeVariables.push({ name, signature: entry.signature, value: serializeRuntimeValue(entry.value) });
    }
  }

  const datasetSelection = window.quickDrawDataset?.selectedClassNames?.() ||
    Array.from(document.querySelectorAll('#classPicker input[type="checkbox"]:checked')).map(input => input.value);

  return {
    format: PROJECT_FORMAT,
    version: PROJECT_VERSION,
    exportedAt: new Date().toISOString(),
    graph: {
      nodes: Array.from(graph.nodes.values()).map(serializeNode),
      connections: graph.connections.map(connection => ({ ...connection }))
    },
    userBlocks: Array.from(USER_BLOCKS.values()).map(definition => JSON.parse(JSON.stringify(definition))),
    runtimeVariables,
    datasetSelection,
    viewport: getWorkspaceViewSnapshot()
  };
}

function safeFileBaseName(text) {
  const cleaned = String(text || '')
    .trim()
    .replace(/[\\/:*?"<>|]+/g, '-')
    .replace(/\s+/g, '-')
    .replace(/^-+|-+$/g, '');
  return cleaned || 'math-network';
}

function exportProject() {
  try {
    downloadJson(JSON.stringify(buildProjectSnapshot()), `${safeFileBaseName('math-network')}-${fileStamp()}.mmlab`);
  } catch (error) {
    console.error(error);
    alert(`내보내기 실패: ${error.message}`);
  }
}

function validateProjectSnapshot(project) {
  if (!project || project.format !== PROJECT_FORMAT) throw new Error('이 프로젝트의 저장 파일이 아닙니다.');
  if (project.version !== PROJECT_VERSION) throw new Error(`지원하지 않는 저장 파일 버전입니다: ${project.version}`);
  if (!Array.isArray(project.graph?.nodes) || !Array.isArray(project.graph?.connections)) throw new Error('노드 또는 연결 정보가 없습니다.');
  if (project.graph.nodes.length > 5000) throw new Error('노드 수가 지나치게 많습니다.');
  if (project.graph.connections.length > 20000) throw new Error('연결 수가 지나치게 많습니다.');
  if (!Array.isArray(project.userBlocks)) throw new Error('사용자 블록 정보가 올바르지 않습니다.');
  if (!Array.isArray(project.runtimeVariables)) throw new Error('변수 정보가 올바르지 않습니다.');
}

function restoreDatasetSelection(names) {
  const wanted = new Set((Array.isArray(names) ? names : []).map(String));
  document.querySelectorAll('#classPicker input[type="checkbox"]').forEach(input => {
    input.checked = wanted.has(input.value);
  });
}

async function restoreProject(project) {
  validateProjectSnapshot(project);

  // Rebuilding a workspace is a long series of edits, and loading the dataset
  // may await. Autosave listens for these so it never stores a half-restored
  // snapshot; the single notify at the end is what schedules the real save.
  window.dispatchEvent(new CustomEvent('workspace-restore-start'));
  try {
    await rebuildWorkspaceFromSnapshot(project);
  } finally {
    window.dispatchEvent(new CustomEvent('workspace-restore-end'));
  }

  notifyWorkspaceChanged();
}

async function rebuildWorkspaceFromSnapshot(project) {
  // Reset visible graph and state before rebuilding it.
  resetWorkspace();

  USER_BLOCKS.clear();
  for (const definition of project.userBlocks) {
    if (!definition?.id || !definition?.name) continue;
    USER_BLOCKS.set(String(definition.id), definition);
  }
  persistUserBlocks();
  renderMyBlocksPalette();

  graph.nodes.clear();
  graph.connections = [];
  nodesLayer.replaceChildren();
  wiresSvg.replaceChildren();

  let largestId = 0;
  const validNodeIds = new Set();
  for (const savedNode of project.graph.nodes) {
    const id = Number(savedNode.id);
    if (!Number.isInteger(id) || id <= 0 || validNodeIds.has(id)) throw new Error('노드 ID가 올바르지 않습니다.');
    const type = String(savedNode.type || '');
    getBlockDef(type); // validates built-in and imported custom block types
    const node = {
      id,
      type,
      x: Number.isFinite(Number(savedNode.x)) ? Number(savedNode.x) : 20,
      y: Number.isFinite(Number(savedNode.y)) ? Number(savedNode.y) : 20,
      params: savedNode.params && typeof savedNode.params === 'object' ? JSON.parse(JSON.stringify(savedNode.params)) : {}
    };
    graph.nodes.set(id, node);
    validNodeIds.add(id);
    largestId = Math.max(largestId, id);
    renderNode(node);
  }

  graph.connections = project.graph.connections
    .map(connection => ({
      from: Number(connection.from),
      to: Number(connection.to),
      inputIndex: Number(connection.inputIndex)
    }))
    .filter(connection => validNodeIds.has(connection.from) && validNodeIds.has(connection.to) && Number.isInteger(connection.inputIndex) && connection.inputIndex >= 0);

  nextNodeId = largestId + 1;
  selectedNodeId = null;
  pendingOutput = null;

  RUNTIME_VARIABLES.clear();
  for (const saved of project.runtimeVariables) {
    const name = String(saved.name || '');
    if (!name) continue;
    RUNTIME_VARIABLES.set(name, {
      signature: String(saved.signature || ''),
      value: deserializeRuntimeValue(saved.value)
    });
  }

  renderInspector();
  syncWorkspaceState();
  updateWires();
  invalidatePreviews();
  restoreWorkspaceViewSnapshot(project?.viewport);

  restoreDatasetSelection(project.datasetSelection);
  if (project.datasetSelection?.length && window.quickDrawDataset?.loadSelectedClasses) {
    await window.quickDrawDataset.loadSelectedClasses();
  }
}

async function importProjectFile(file) {
  if (!file) return;
  if (file.size > MAX_PROJECT_FILE_BYTES) throw new Error('저장 파일이 50MB보다 큽니다.');
  const text = await file.text();
  let project;
  try { project = JSON.parse(text); }
  catch { throw new Error('파일이 올바른 JSON 형식이 아닙니다.'); }
  await restoreProject(project);
}

// ---------- .mmlweights: trained parameters only ----------
//
// A trainable parameter is identified by the variable named on a 미분 block that
// is also the target of at least one 값 바꾸기. This works for both direct SGD
// and minibatch training: in a minibatch graph the derivative flows into an
// accumulator such as gW while the actual parameter W is updated from it later,
// so following the derivative into 값 바꾸기 would save the wrong variable.

function collectSetVariableTargets() {
  const names = new Set();
  for (const node of graph.nodes.values()) {
    if (getBlockDef(node.type).special !== 'setVariable') continue;
    const name = String(node.params.variable || '').trim();
    if (name) names.add(name);
  }
  return names;
}

function collectTrainableVariableNames() {
  const updateTargets = collectSetVariableTargets();
  const names = new Set();

  for (const node of graph.nodes.values()) {
    if (getBlockDef(node.type).special !== 'derivative') continue;
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
    return { name, value: serializeRuntimeValue(readRuntimeVariable(variableNode)) };
  });

  return {
    format: WEIGHTS_FORMAT,
    version: WEIGHTS_VERSION,
    exportedAt: new Date().toISOString(),
    classes: window.quickDrawDataset?.getLoadedClassNames?.() || [],
    variables
  };
}

function exportWeights() {
  if (evaluateBtn?.disabled) {
    alert('계산이 끝난 뒤 가중치를 저장하세요.');
    return;
  }
  try {
    const snapshot = buildWeightsSnapshot();
    downloadJson(JSON.stringify(snapshot), `math-weights-${fileStamp()}.mmlweights`);
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

// ---------- buttons ----------

exportProjectBtn?.addEventListener('click', exportProject);
importProjectBtn?.addEventListener('click', () => importProjectInput?.click());
importProjectInput?.addEventListener('change', async () => {
  const file = importProjectInput.files?.[0];
  if (!file) return;
  importProjectBtn.disabled = true;
  exportProjectBtn.disabled = true;
  try {
    await importProjectFile(file);
    workspaceStatus.textContent = `${graph.nodes.size}개 블록 · 불러옴`;
  } catch (error) {
    console.error(error);
    alert(`불러오기 실패: ${error.message}`);
  } finally {
    importProjectInput.value = '';
    importProjectBtn.disabled = false;
    exportProjectBtn.disabled = false;
  }
});

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
