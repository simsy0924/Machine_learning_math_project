// Save and restore a complete math-block workspace without embedding the Quick Draw dataset.

const PROJECT_FORMAT = 'machine-learning-math-project';
const PROJECT_VERSION = 1;
const MAX_PROJECT_FILE_BYTES = 50 * 1024 * 1024;

const exportProjectBtn = document.getElementById('exportProjectBtn');
const importProjectBtn = document.getElementById('importProjectBtn');
const importProjectInput = document.getElementById('importProjectInput');

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
    return {
      type: 'float32',
      shape: [...array.shape],
      data: bytesToBase64(bytes)
    };
  }
  throw new Error('저장할 수 없는 변수 값이 있습니다.');
}

function deserializeRuntimeValue(saved) {
  if (saved?.type === 'number') return Number(saved.value);
  if (saved?.type === 'float32') {
    if (!Array.isArray(saved.shape) || !saved.shape.length) throw new Error('배열 모양 정보가 없습니다.');
    const bytes = base64ToBytes(String(saved.data || ''));
    if (bytes.byteLength % 4 !== 0) throw new Error('Float32 데이터 크기가 올바르지 않습니다.');
    const copied = bytes.slice().buffer;
    const data = new Float32Array(copied);
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

function buildProjectSnapshot() {
  const runtimeVariables = [];
  if (typeof RUNTIME_VARIABLES !== 'undefined') {
    for (const [name, entry] of RUNTIME_VARIABLES.entries()) {
      runtimeVariables.push({
        name,
        signature: entry.signature,
        value: serializeRuntimeValue(entry.value)
      });
    }
  }

  const datasetSelection = window.quickDrawDataset?.selectedClassNames?.() || [];

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
    datasetSelection
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
    const snapshot = buildProjectSnapshot();
    const json = JSON.stringify(snapshot);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    const now = new Date();
    const stamp = [
      now.getFullYear(),
      String(now.getMonth() + 1).padStart(2, '0'),
      String(now.getDate()).padStart(2, '0'),
      '-',
      String(now.getHours()).padStart(2, '0'),
      String(now.getMinutes()).padStart(2, '0')
    ].join('');
    anchor.href = url;
    anchor.download = `${safeFileBaseName('math-network')}-${stamp}.mmlab`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
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

  if (typeof RUNTIME_VARIABLES !== 'undefined') {
    RUNTIME_VARIABLES.clear();
    for (const saved of project.runtimeVariables) {
      const name = String(saved.name || '');
      if (!name) continue;
      RUNTIME_VARIABLES.set(name, {
        signature: String(saved.signature || ''),
        value: deserializeRuntimeValue(saved.value)
      });
    }
  }

  renderInspector();
  syncWorkspaceState();
  updateWires();
  invalidatePreviews();

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
