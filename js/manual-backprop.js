// Manual backpropagation authored with the same math blocks as the forward pass.
//
// There is deliberately no graph-level automatic differentiation here. A block
// type may have a user-authored backward definition that receives:
//   - the block's original input values
//   - the block's forward output value
//   - g = dL/d(output), supplied explicitly by the user
// and produces one gradient value for each input.
//
// In the normal workspace a block instance can be switched to "역전파 실행".
// That instance reads g from a named 변수 block and writes each input gradient to
// named 변수 blocks immediately. The user decides the reverse execution order
// (usually with 둘 다 계산), so chain-rule wiring remains visible and manual.

const MANUAL_BACKPROP_STORAGE_KEY = 'machine-learning-math-project.manual-backprop.v1';
const MANUAL_BACKPROP_SOURCE_TYPE = '__manualBackpropSource';
const MANUAL_BACKPROP_UPSTREAM_ID = -1000001;
const MANUAL_BACKPROP_OUTPUT_ID = -1000002;
const MANUAL_BACKPROP_VERSION = 1;

const MANUAL_BACKPROP_BUILTINS = new Map();
let manualBackpropEditorState = null;
let manualBackpropEditorToolbar = null;
let manualBackpropEditorTitle = null;

function manualBackpropInputSourceId(index) {
  return -(index + 1);
}

function isManualBackpropWorkspaceEditing() {
  return Boolean(manualBackpropEditorState);
}

function cloneManualJson(value) {
  return JSON.parse(JSON.stringify(value));
}

// ---------- retire automatic differentiation ----------

// The old 미분 block and VJP metadata are intentionally removed at runtime too,
// not merely hidden from the palette. The old autodiff modules are no longer
// loaded by index.html on this branch.
delete BLOCKS.derivative;
for (const definition of Object.values(BLOCKS)) delete definition.vjp;
GRADIENT_STRATEGIES.length = 0;
if (typeof UNSUPPORTED_IN_USER_BLOCK !== 'undefined') UNSUPPORTED_IN_USER_BLOCK.delete('derivative');

// A general linear-algebra operation that is needed to write matrix-vector
// backprop by hand. This is not a backward-specific block: it simply computes
// A^T x and is useful anywhere in the forward graph too.
if (!BLOCKS.transposeMatvec) {
  BLOCKS.transposeMatvec = {
    title: '전치 행렬 × 벡터',
    kind: 'operation',
    inputs: ['행렬', '벡터'],
    description: '행렬 A와 벡터 x로 Aᵀx를 계산한다.',
    formula: () => 'Aᵀx',
    compute: (node, [matrix, vector]) => transposeMatrixVectorValues(matrix, vector)
  };
}

// Editor-only sources. They never enter a saved normal workspace graph.
BLOCKS[MANUAL_BACKPROP_SOURCE_TYPE] = {
  title: '역전파 입력',
  kind: 'source',
  inputs: [],
  description: '역전파 정의 편집기에서만 보이는 값입니다.',
  formula: node => {
    const role = String(node.params.role || 'input');
    if (role === 'upstream') return 'g = ∂L/∂y';
    if (role === 'output') return 'y';
    return String(node.params.label || 'x');
  },
  controls: [],
  compute: () => 0
};

const FORWARD_GET_BLOCK_DEF = getBlockDef;

// ---------- persistence ----------

function loadManualBackpropBuiltins() {
  MANUAL_BACKPROP_BUILTINS.clear();
  try {
    const raw = localStorage.getItem(MANUAL_BACKPROP_STORAGE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    if (!Array.isArray(saved)) return;
    for (const entry of saved) {
      if (!entry?.type || !entry?.definition) continue;
      MANUAL_BACKPROP_BUILTINS.set(String(entry.type), entry.definition);
    }
  } catch (error) {
    console.warn('수동 역전파 정의 불러오기 실패', error);
  }
}

function persistManualBackpropBuiltins() {
  try {
    const saved = Array.from(MANUAL_BACKPROP_BUILTINS, ([type, definition]) => ({ type, definition }));
    localStorage.setItem(MANUAL_BACKPROP_STORAGE_KEY, JSON.stringify(saved));
  } catch (error) {
    console.warn('수동 역전파 정의 저장 실패', error);
  }
}

loadManualBackpropBuiltins();

function manualBackpropDefinitionForType(type) {
  if (String(type).startsWith('custom:')) {
    return USER_BLOCKS.get(String(type).slice(7))?.manualBackprop || null;
  }
  return MANUAL_BACKPROP_BUILTINS.get(String(type)) || null;
}

function saveManualBackpropDefinition(type, definition) {
  const key = String(type);
  if (key.startsWith('custom:')) {
    const custom = USER_BLOCKS.get(key.slice(7));
    if (!custom) throw new Error('사용자 블록을 찾지 못했습니다.');
    // Forward structure did not change, so keep the same object and compiled
    // forward plan. The manual definition is independent metadata.
    custom.manualBackprop = definition;
    persistUserBlocks();
    renderMyBlocksPalette();
    return;
  }

  MANUAL_BACKPROP_BUILTINS.set(key, definition);
  persistManualBackpropBuiltins();
}

// Include built-in manual definitions in .mmlab/autosave snapshots. Custom block
// definitions already travel inside userBlocks because manualBackprop is stored
// on the custom definition itself.
const buildProjectSnapshotWithoutManualBackprop = buildProjectSnapshot;
buildProjectSnapshot = function buildProjectSnapshotWithManualBackprop(options = {}) {
  const snapshot = buildProjectSnapshotWithoutManualBackprop(options);
  snapshot.manualBackprop = {
    version: MANUAL_BACKPROP_VERSION,
    builtins: Array.from(MANUAL_BACKPROP_BUILTINS, ([type, definition]) => ({
      type,
      definition: cloneManualJson(definition)
    }))
  };
  return snapshot;
};

const rebuildWorkspaceWithoutManualBackprop = rebuildWorkspaceFromSnapshot;
rebuildWorkspaceFromSnapshot = async function rebuildWorkspaceWithManualBackprop(project) {
  const saved = project?.manualBackprop;
  if (saved?.version === MANUAL_BACKPROP_VERSION && Array.isArray(saved.builtins)) {
    MANUAL_BACKPROP_BUILTINS.clear();
    for (const entry of saved.builtins) {
      if (!entry?.type || !entry?.definition) continue;
      MANUAL_BACKPROP_BUILTINS.set(String(entry.type), cloneManualJson(entry.definition));
    }
    persistManualBackpropBuiltins();
  }
  return rebuildWorkspaceWithoutManualBackprop(project);
};

// Autosave already knows to suspend itself while a 사용자 블록 editor owns the
// temporary graph. Extend that same check to this editor so its temporary graph
// can never overwrite the real workspace snapshot.
const userBlockWorkspaceEditingOnly = isUserBlockWorkspaceEditing;
isUserBlockWorkspaceEditing = function isAnyDefinitionWorkspaceEditing() {
  return userBlockWorkspaceEditingOnly() || isManualBackpropWorkspaceEditing();
};
window.isUserBlockWorkspaceEditing = isUserBlockWorkspaceEditing;
window.isManualBackpropWorkspaceEditing = isManualBackpropWorkspaceEditing;

// A custom block referenced from a saved backward definition is still in use and
// must not be deleted. Extend the existing dependency check with those graphs.
const userBlockUsageWithoutManualBackprop = userBlockUsage;
userBlockUsage = function userBlockUsageWithManualBackprop(customId) {
  const usage = userBlockUsageWithoutManualBackprop(customId);
  const type = `custom:${customId}`;
  const reasons = new Set(usage.parentBlocks || []);

  for (const definition of USER_BLOCKS.values()) {
    if (definition.id === customId) continue;
    if ((definition.manualBackprop?.nodes || []).some(node => node.type === type)) {
      reasons.add(`${definition.name || definition.id}의 역전파 정의`);
    }
  }

  for (const [ownerType, definition] of MANUAL_BACKPROP_BUILTINS) {
    if ((definition.nodes || []).some(node => node.type === type)) {
      let title = ownerType;
      try { title = FORWARD_GET_BLOCK_DEF(ownerType).title; } catch { /* raw type */ }
      reasons.add(`${title}의 역전파 정의`);
    }
  }

  return { ...usage, parentBlocks: Array.from(reasons) };
};

// ---------- which blocks can own a manual backward definition ----------

function manualBackpropEligible(type, definition = null) {
  const key = String(type || '');
  if (!key || key === MANUAL_BACKPROP_SOURCE_TYPE) return false;

  const def = definition || FORWARD_GET_BLOCK_DEF(key);
  if (!Array.isArray(def.inputs) || !def.inputs.length) return false;
  if (def.special) return false;
  if (key === 'sequence') return false;
  return def.kind === 'operation' || def.kind === 'transform' || def.kind === 'custom' || def.kind === 'sink';
}

function manualBackpropInternalBlockAllowed(type) {
  const key = String(type || '');
  if (key === MANUAL_BACKPROP_SOURCE_TYPE) return true;
  if (key === 'number') return true;
  if (key === 'constantVector' || key === 'randomVector' || key === 'matrix') return true;
  let def;
  try { def = FORWARD_GET_BLOCK_DEF(key); } catch { return false; }
  if (def.special) return false;
  return def.kind === 'operation' || def.kind === 'transform' || def.kind === 'custom' || def.kind === 'sink';
}

// ---------- execute an explicitly authored backward graph ----------

function manualBackpropSourceValue(id, inputValues, forwardOutput, upstream) {
  if (id === MANUAL_BACKPROP_UPSTREAM_ID) return upstream;
  if (id === MANUAL_BACKPROP_OUTPUT_ID) return forwardOutput;
  if (id < 0) {
    const index = -id - 1;
    if (index >= 0 && index < inputValues.length) return inputValues[index];
  }
  throw new Error('역전파 정의의 입력 소스를 찾지 못했습니다.');
}

function sanitizedManualNode(node) {
  const params = cloneManualJson(node.params || {});
  delete params.manualBackpropMode;
  delete params.manualBackpropUpstream;
  for (const key of Object.keys(params)) {
    if (key.startsWith('manualBackpropGradient')) delete params[key];
  }
  return { ...node, params };
}

function evaluateManualBackpropDefinition(definition, inputValues, forwardOutput, upstream) {
  if (!definition || !Array.isArray(definition.nodes) || !Array.isArray(definition.connections)) {
    throw new Error('이 블록의 역전파 정의가 올바르지 않습니다.');
  }

  const outputIds = Array.isArray(definition.gradientOutputNodeIds)
    ? definition.gradientOutputNodeIds
    : [];
  if (outputIds.length !== inputValues.length || outputIds.some(id => id == null)) {
    throw new Error('모든 입력의 기울기 출력이 지정되지 않았습니다.');
  }

  const nodes = new Map();
  for (const saved of definition.nodes) nodes.set(Number(saved.id), sanitizedManualNode(saved));

  const connectionByInput = new Map();
  for (const connection of definition.connections) {
    const key = `${Number(connection.to)}:${Number(connection.inputIndex)}`;
    if (!connectionByInput.has(key)) connectionByInput.set(key, Number(connection.from));
  }

  const memo = new Map();
  const visiting = new Set();

  const evaluate = id => {
    const numericId = Number(id);
    if (numericId < 0) return manualBackpropSourceValue(numericId, inputValues, forwardOutput, upstream);
    if (memo.has(numericId)) return memo.get(numericId);
    if (visiting.has(numericId)) throw new Error('역전파 정의에 순환 연결이 있습니다.');

    const node = nodes.get(numericId);
    if (!node) throw new Error('역전파 정의의 계산 블록을 찾지 못했습니다.');
    const def = getBlockDef(node.type);
    if (!manualBackpropInternalBlockAllowed(node.type)) {
      throw new Error(`'${def.title}' 블록은 역전파 정의 안에서 사용할 수 없습니다.`);
    }

    visiting.add(numericId);
    try {
      const inputs = [];
      for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
        const from = connectionByInput.get(`${numericId}:${inputIndex}`);
        if (from == null) throw new Error(`역전파 정의에서 '${def.inputs[inputIndex]}' 입력이 비어 있습니다.`);
        inputs.push(evaluate(from));
      }
      const value = def.compute(node, inputs);
      memo.set(numericId, value);
      return value;
    } finally {
      visiting.delete(numericId);
    }
  };

  return outputIds.map(evaluate);
}

function runtimeVariableValueByName(name) {
  const variableNode = findVariableNode(String(name || ''));
  if (!variableNode) throw new Error(`'${name}'이라는 변수 블록을 찾지 못했습니다.`);
  return readRuntimeVariable(variableNode);
}

function executeManualBackpropNode(type, baseDefinition, node, inputs) {
  const definition = manualBackpropDefinitionForType(type);
  if (!definition) {
    throw new Error(`'${baseDefinition.title}' 블록의 역전파 정의가 없습니다. 먼저 인스펙터에서 직접 정의해 주세요.`);
  }

  const forwardOutput = baseDefinition.compute(node, inputs);
  const upstreamName = String(node.params.manualBackpropUpstream || 'g').trim() || 'g';
  const upstream = runtimeVariableValueByName(upstreamName);
  const gradients = evaluateManualBackpropDefinition(definition, inputs, forwardOutput, upstream);

  for (let i = 0; i < gradients.length; i++) {
    const targetName = String(node.params[`manualBackpropGradient${i}`] || '').trim();
    if (!targetName) continue;
    // Gradient scratch variables must be readable by the very next backward
    // block in the same repeat iteration, so these writes are intentionally
    // immediate. Weight updates done with 값 바꾸기 keep their existing
    // end-of-iteration commit semantics.
    writeRuntimeVariable(targetName, gradients[i], true);
  }

  // The node's ordinary single output is only for sequencing/inspection. All
  // input gradients are written to the explicitly named gradient variables.
  return gradients[0] ?? upstream;
}

// Keep the original catalogue as the forward definition. This wrapper does not
// infer or propagate anything: it merely changes one explicitly configured block
// instance from forward evaluation to execution of its saved manual formula.
getBlockDef = function getBlockDefWithManualBackprop(type) {
  const base = FORWARD_GET_BLOCK_DEF(type);
  if (!manualBackpropEligible(type, base)) return base;

  return {
    ...base,
    description: `${base.description} 역전파 식은 자동 생성되지 않으며 사용자가 직접 정의할 수 있다.`,
    formula: node => node?.params?.manualBackpropMode === 'backward'
      ? `manual backward: ${node.params.manualBackpropUpstream || 'g'} → 입력 기울기`
      : base.formula(node),
    compute: (node, inputs) => node?.params?.manualBackpropMode === 'backward'
      ? executeManualBackpropNode(type, base, node, inputs)
      : base.compute(node, inputs)
  };
};

// ---------- backward-definition workspace editor ----------

function manualBackpropTargetInfo(type) {
  const def = FORWARD_GET_BLOCK_DEF(type);
  return {
    type: String(type),
    title: def.title,
    inputLabels: [...def.inputs]
  };
}

function manualBackpropSourceNodes(info) {
  const nodes = [];
  info.inputLabels.forEach((label, index) => {
    nodes.push({
      id: manualBackpropInputSourceId(index),
      type: MANUAL_BACKPROP_SOURCE_TYPE,
      x: 30,
      y: 45 + index * 120,
      params: { role: 'input', sourceIndex: index, label: String(label) }
    });
  });
  nodes.push({
    id: MANUAL_BACKPROP_UPSTREAM_ID,
    type: MANUAL_BACKPROP_SOURCE_TYPE,
    x: 30,
    y: 65 + info.inputLabels.length * 120,
    params: { role: 'upstream', label: 'g' }
  });
  nodes.push({
    id: MANUAL_BACKPROP_OUTPUT_ID,
    type: MANUAL_BACKPROP_SOURCE_TYPE,
    x: 30,
    y: 165 + info.inputLabels.length * 120,
    params: { role: 'output', label: '순전파 출력 y' }
  });
  return nodes;
}

function buildManualBackpropEditorGraph(info, saved) {
  const nodes = new Map();
  let largestId = 0;

  for (const source of manualBackpropSourceNodes(info)) nodes.set(source.id, source);

  for (const raw of saved?.nodes || []) {
    const id = Number(raw.id);
    if (!Number.isInteger(id) || id <= 0 || nodes.has(id)) continue;
    const type = String(raw.type || '');
    if (!manualBackpropInternalBlockAllowed(type)) continue;
    const node = sanitizedManualNode({
      id,
      type,
      x: Number.isFinite(Number(raw.x)) ? Number(raw.x) : 280,
      y: Number.isFinite(Number(raw.y)) ? Number(raw.y) : 60,
      params: raw.params && typeof raw.params === 'object' ? raw.params : {}
    });
    nodes.set(id, node);
    largestId = Math.max(largestId, id);
  }

  const validIds = new Set(nodes.keys());
  const connections = [];
  for (const raw of saved?.connections || []) {
    const from = Number(raw.from);
    const to = Number(raw.to);
    const inputIndex = Number(raw.inputIndex);
    if (!validIds.has(from) || !validIds.has(to) || to < 0) continue;
    if (!Number.isInteger(inputIndex) || inputIndex < 0) continue;
    connections.push({ from, to, inputIndex });
  }

  const gradientOutputNodeIds = new Array(info.inputLabels.length).fill(null);
  const savedOutputs = Array.isArray(saved?.gradientOutputNodeIds) ? saved.gradientOutputNodeIds : [];
  for (let i = 0; i < gradientOutputNodeIds.length; i++) {
    const id = Number(savedOutputs[i]);
    if (validIds.has(id)) gradientOutputNodeIds[i] = id;
  }

  return { nodes, connections, nextId: Math.max(1, largestId + 1), gradientOutputNodeIds };
}

function ensureManualBackpropEditorToolbar() {
  if (manualBackpropEditorToolbar) return;
  manualBackpropEditorToolbar = document.createElement('div');
  manualBackpropEditorToolbar.className = 'manual-backprop-editor-toolbar';
  manualBackpropEditorToolbar.hidden = true;
  manualBackpropEditorToolbar.innerHTML = `
    <span class="pill manual-backprop-editor-badge">수동 역전파 정의</span>
    <strong id="manualBackpropEditorTitle"></strong>
    <span class="muted manual-backprop-editor-help">g와 입력/출력 값을 일반 수학 블록에 직접 연결하세요.</span>
    <button type="button" id="manualBackpropEditorSave" class="primary">저장하고 돌아가기</button>
    <button type="button" id="manualBackpropEditorCancel" class="ghost">취소</button>`;
  document.querySelector('.workspace-head')?.appendChild(manualBackpropEditorToolbar);
  manualBackpropEditorTitle = document.getElementById('manualBackpropEditorTitle');
  document.getElementById('manualBackpropEditorSave')?.addEventListener('click', saveManualBackpropEditor);
  document.getElementById('manualBackpropEditorCancel')?.addEventListener('click', cancelManualBackpropEditor);
}

function setManualBackpropEditorControlsDisabled(disabled, state = manualBackpropEditorState) {
  setEditingControlsDisabled(disabled, state);
  const ids = ['groupSelectBtn', 'createGroupBtn', 'cancelGroupBtn'];
  if (disabled) {
    state.manualDisabledStates = new Map();
    for (const id of ids) {
      const element = document.getElementById(id);
      if (!element) continue;
      state.manualDisabledStates.set(id, element.disabled);
      element.disabled = true;
    }
  } else {
    for (const [id, oldValue] of state?.manualDisabledStates || []) {
      const element = document.getElementById(id);
      if (element) element.disabled = oldValue;
    }
  }
}

function decorateManualBackpropEditorNodes() {
  if (!manualBackpropEditorState) return;
  const outputs = new Set(manualBackpropEditorState.gradientOutputNodeIds.filter(id => id != null));
  nodesLayer.querySelectorAll('.node').forEach(element => {
    const id = Number(element.dataset.nodeId);
    element.classList.toggle('manual-backprop-source', id < 0);
    element.classList.toggle('manual-backprop-gradient-output', outputs.has(id));
  });
}

function redrawManualBackpropEditorWorkspace() {
  renderWorkspaceGraph();
  pendingOutput = null;
  selectedConnection = null;
  selectedNodeId = null;
  renderInspector();
  syncWorkspaceState();
  updateWires();
  decorateManualBackpropEditorNodes();
}

function openManualBackpropEditor(node) {
  if (!node || manualBackpropEditorState || userBlockWorkspaceEditingOnly()) return;
  const baseDef = FORWARD_GET_BLOCK_DEF(node.type);
  if (!manualBackpropEligible(node.type, baseDef)) return;
  if (evaluateBtn.disabled) {
    alert('계산이 끝난 뒤 역전파 정의를 편집해 주세요.');
    return;
  }

  cancelGroupSelection();
  cancelConnection();

  const info = manualBackpropTargetInfo(node.type);
  const saved = manualBackpropDefinitionForType(node.type);
  const built = buildManualBackpropEditorGraph(info, saved);

  manualBackpropEditorState = {
    targetType: node.type,
    targetInfo: info,
    gradientOutputNodeIds: built.gradientOutputNodeIds,
    outerNodes: graph.nodes,
    outerConnections: graph.connections,
    outerNextNodeId: nextNodeId,
    outerSelectedNodeId: selectedNodeId,
    outerViewport: getWorkspaceViewSnapshot(),
    disabledStates: new Map(),
    manualDisabledStates: new Map()
  };

  graph.nodes = built.nodes;
  graph.connections = built.connections;
  nextNodeId = built.nextId;

  ensureManualBackpropEditorToolbar();
  manualBackpropEditorTitle.textContent = `${info.title} · 입력 ${info.inputLabels.length}개`;
  manualBackpropEditorToolbar.hidden = false;
  setManualBackpropEditorControlsDisabled(true);
  redrawManualBackpropEditorWorkspace();
  restoreWorkspaceViewSnapshot(saved?.editorViewport || { x: 0, y: 0, zoom: 1 });
  connectionHint.textContent = '자동미분은 없습니다. g에서 시작해 각 입력의 기울기를 직접 계산하세요.';
  workspaceStatus.textContent = '수동 역전파 식 편집 중';
}

function collectManualBackpropDefinition() {
  if (!manualBackpropEditorState) throw new Error('역전파 정의를 편집 중이 아닙니다.');
  const info = manualBackpropEditorState.targetInfo;

  const internalNodes = [...graph.nodes.values()].filter(node => node.id > 0);
  const validIds = new Set(graph.nodes.keys());

  for (const node of internalNodes) {
    if (!manualBackpropInternalBlockAllowed(node.type)) {
      let title = node.type;
      try { title = FORWARD_GET_BLOCK_DEF(node.type).title; } catch { /* raw type */ }
      throw new Error(`'${title}' 블록은 역전파 정의 안에 저장할 수 없습니다.`);
    }
    const def = getBlockDef(node.type);
    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const matches = graph.connections.filter(c => c.to === node.id && c.inputIndex === inputIndex);
      if (matches.length !== 1) {
        throw new Error(`'${def.title}' 블록의 '${def.inputs[inputIndex]}' 입력을 정확히 하나 연결해 주세요.`);
      }
      if (!validIds.has(matches[0].from)) throw new Error('올바르지 않은 역전파 연결이 있습니다.');
    }
  }

  const outputs = manualBackpropEditorState.gradientOutputNodeIds;
  if (outputs.length !== info.inputLabels.length || outputs.some(id => id == null || !validIds.has(id))) {
    throw new Error('각 입력마다 기울기 출력 블록을 하나씩 지정해 주세요.');
  }

  const nodes = internalNodes.map(node => ({
    id: node.id,
    type: node.type,
    x: node.x,
    y: node.y,
    params: sanitizedManualNode(node).params
  }));

  const internalTargetIds = new Set(internalNodes.map(node => node.id));
  const connections = graph.connections
    .filter(connection => internalTargetIds.has(connection.to) && validIds.has(connection.from))
    .map(connection => ({
      from: connection.from,
      to: connection.to,
      inputIndex: connection.inputIndex
    }));

  return {
    version: MANUAL_BACKPROP_VERSION,
    inputCount: info.inputLabels.length,
    nodes,
    connections,
    gradientOutputNodeIds: [...outputs],
    editorViewport: getWorkspaceViewSnapshot()
  };
}

function restoreOuterWorkspaceFromManualBackprop(state) {
  graph.nodes = state.outerNodes;
  graph.connections = state.outerConnections;
  nextNodeId = state.outerNextNodeId;
  selectedNodeId = state.outerSelectedNodeId;
  pendingOutput = null;
  selectedConnection = null;

  renderWorkspaceGraph();
  setManualBackpropEditorControlsDisabled(false, state);
  if (manualBackpropEditorToolbar) manualBackpropEditorToolbar.hidden = true;
  syncWorkspaceState();
  updateWires();
  restoreWorkspaceViewSnapshot(state.outerViewport);
  renderInspector();
}

function saveManualBackpropEditor() {
  if (!manualBackpropEditorState) return;
  let definition;
  try {
    definition = collectManualBackpropDefinition();
  } catch (error) {
    alert(`역전파 정의를 저장할 수 없습니다.\n${error.message}`);
    return;
  }

  const state = manualBackpropEditorState;
  try {
    saveManualBackpropDefinition(state.targetType, definition);
  } catch (error) {
    alert(`역전파 정의 저장 실패: ${error.message}`);
    return;
  }

  manualBackpropEditorState = null;
  restoreOuterWorkspaceFromManualBackprop(state);
  notifyWorkspaceChanged();
}

function cancelManualBackpropEditor() {
  if (!manualBackpropEditorState) return;
  if (!confirm('역전파 정의 편집을 취소할까요?\n편집 중 바꾼 식은 저장되지 않습니다.')) return;
  const state = manualBackpropEditorState;
  manualBackpropEditorState = null;
  restoreOuterWorkspaceFromManualBackprop(state);
}

function setSelectedAsManualGradientOutput(inputIndex) {
  if (!manualBackpropEditorState || selectedNodeId == null) return;
  if (!graph.nodes.has(selectedNodeId)) return;
  manualBackpropEditorState.gradientOutputNodeIds[inputIndex] = selectedNodeId;
  decorateManualBackpropEditorNodes();
  renderInspector();
}

// ---------- inspector UI ----------

function appendManualControl(labelText, control) {
  const row = document.createElement('div');
  row.className = 'control-row manual-backprop-control-row';
  const label = document.createElement('label');
  label.textContent = labelText;
  row.append(label, control);
  inspectorControls.appendChild(row);
}

function defaultGradientVariableName(label, index) {
  const cleaned = String(label || '').replace(/[^A-Za-z0-9가-힣_]/g, '');
  return `d${cleaned || index + 1}`;
}

function renderManualBackpropRuntimeControls(node, def, hasDefinition) {
  if (node.params.manualBackpropMode === 'backward' && !hasDefinition) {
    node.params.manualBackpropMode = 'forward';
  }

  const mode = document.createElement('select');
  mode.append(new Option('순전파', 'forward'));
  if (hasDefinition) mode.append(new Option('역전파 실행', 'backward'));
  mode.value = node.params.manualBackpropMode || 'forward';
  mode.addEventListener('input', () => {
    node.params.manualBackpropMode = mode.value;
    invalidatePreviews();
    notifyWorkspaceChanged();
    renderInspector();
  });
  appendManualControl('실행 방식', mode);

  if ((node.params.manualBackpropMode || 'forward') !== 'backward') return;

  if (!node.params.manualBackpropUpstream) node.params.manualBackpropUpstream = 'g';
  const upstream = document.createElement('input');
  upstream.type = 'text';
  upstream.value = node.params.manualBackpropUpstream;
  upstream.addEventListener('input', () => {
    node.params.manualBackpropUpstream = upstream.value;
    invalidatePreviews();
    notifyWorkspaceChanged();
  });
  appendManualControl('출력 기울기 변수 g', upstream);

  def.inputs.forEach((label, index) => {
    const key = `manualBackpropGradient${index}`;
    if (node.params[key] == null) node.params[key] = defaultGradientVariableName(label, index);
    const input = document.createElement('input');
    input.type = 'text';
    input.value = node.params[key];
    input.placeholder = '비우면 저장하지 않음';
    input.addEventListener('input', () => {
      node.params[key] = input.value;
      invalidatePreviews();
      notifyWorkspaceChanged();
    });
    appendManualControl(`${label} 기울기 → 변수`, input);
  });

  const note = document.createElement('p');
  note.className = 'muted manual-backprop-runtime-note';
  note.textContent = '프로그램은 역전파 순서를 자동으로 찾지 않습니다. 뒤쪽 블록부터 직접 실행하고, 위 변수 이름으로 기울기를 이어 주세요.';
  inspectorControls.appendChild(note);
}

INSPECTOR_EXTENSIONS.push((node, currentDef) => {
  if (inspectorContent.hidden) return;

  if (manualBackpropEditorState) {
    // The 사용자 블록 extension ran earlier; nested definition editors would
    // swap the same global graph twice, so hide that action while editing a
    // backward formula.
    inspectorControls.querySelector('.user-block-open-button')?.remove();

    if (node.id < 0) {
      const note = document.createElement('p');
      note.className = 'muted manual-backprop-source-note';
      if (node.id === MANUAL_BACKPROP_UPSTREAM_ID) {
        note.textContent = '출력에서 들어오는 기울기 g입니다. 체인 룰의 시작점으로 사용하세요.';
      } else if (node.id === MANUAL_BACKPROP_OUTPUT_ID) {
        note.textContent = '이 블록의 순전파 출력값 y입니다. exp처럼 출력값으로 미분식을 쓰고 싶을 때 사용하세요.';
      } else {
        note.textContent = '순전파 때 이 블록으로 들어왔던 실제 입력값입니다.';
      }
      inspectorControls.appendChild(note);
    }

    const heading = document.createElement('p');
    heading.className = 'manual-backprop-output-heading';
    heading.textContent = '이 값을 어떤 입력의 기울기로 사용할까요?';
    inspectorControls.appendChild(heading);

    manualBackpropEditorState.targetInfo.inputLabels.forEach((label, inputIndex) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'ghost manual-backprop-output-button';
      const active = manualBackpropEditorState.gradientOutputNodeIds[inputIndex] === node.id;
      button.textContent = active ? `${label}: 현재 기울기 출력` : `${label}의 기울기로 지정`;
      button.disabled = active;
      button.addEventListener('click', () => setSelectedAsManualGradientOutput(inputIndex));
      inspectorControls.appendChild(button);
    });
    return;
  }

  if (userBlockWorkspaceEditingOnly()) return;
  const baseDef = FORWARD_GET_BLOCK_DEF(node.type);
  if (!manualBackpropEligible(node.type, baseDef)) return;

  const definition = manualBackpropDefinitionForType(node.type);
  const section = document.createElement('div');
  section.className = 'manual-backprop-inspector-section';

  const status = document.createElement('p');
  status.className = 'muted manual-backprop-status';
  status.textContent = definition
    ? '역전파: 사용자가 정의한 식이 저장되어 있습니다.'
    : '역전파: 아직 정의되지 않았습니다. 자동미분은 사용하지 않습니다.';

  const edit = document.createElement('button');
  edit.type = 'button';
  edit.className = definition ? 'ghost manual-backprop-edit-button' : 'primary manual-backprop-edit-button';
  edit.textContent = definition ? '역전파 정의 편집' : '역전파 직접 정의';
  edit.addEventListener('click', () => openManualBackpropEditor(node));

  section.append(status, edit);
  inspectorControls.prepend(section);
  renderManualBackpropRuntimeControls(node, baseDef, Boolean(definition));
});

// Protect editor-only source nodes and keep stateful/data blocks out of a saved
// derivative formula.
deleteNodeBtn.addEventListener('click', event => {
  if (!manualBackpropEditorState || selectedNodeId == null || selectedNodeId >= 0) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  alert('입력값, 순전파 출력 y, 출력 기울기 g는 역전파 정의의 고정 입력이라 삭제할 수 없습니다.');
}, true);

document.addEventListener('click', event => {
  if (!manualBackpropEditorState) return;
  const button = event.target.closest('[data-block]');
  if (!button) return;
  const type = String(button.dataset.block || '');
  if (manualBackpropInternalBlockAllowed(type)) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  alert('역전파 식에는 숫자/행렬과 일반 수학 연산, 변환, 사용자 블록만 사용할 수 있습니다. 변수·데이터·반복·값 바꾸기는 바깥에 두세요.');
}, true);

// ---------- small visual treatment ----------

const manualBackpropStyle = document.createElement('style');
manualBackpropStyle.textContent = `
  .manual-backprop-editor-toolbar { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
  .manual-backprop-editor-help { font-size:12px; margin-right:auto; }
  .manual-backprop-source { outline:2px dashed rgba(71, 111, 214, .55); }
  .manual-backprop-gradient-output { box-shadow:0 0 0 3px rgba(211, 76, 76, .35); }
  .manual-backprop-inspector-section { display:grid; gap:7px; padding:9px 0; border-bottom:1px solid var(--line, #ddd); margin-bottom:8px; }
  .manual-backprop-status, .manual-backprop-runtime-note, .manual-backprop-source-note { margin:0; }
  .manual-backprop-output-heading { margin:10px 0 4px; font-weight:700; }
  .manual-backprop-output-button { width:100%; margin-top:5px; }
`;
document.head.appendChild(manualBackpropStyle);
