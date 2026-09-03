// 사용자 블록: grouping a selection into a reusable block, the palette entry it
// gets, deleting it, and editing its internals inside the normal workspace.
//
// Opening a grouped block temporarily swaps the live workspace graph with the
// block's saved internal graph. Because the normal renderer, ports, inspector,
// pan/zoom and editing handlers are reused, the inside looks and behaves exactly
// like the rest of the app instead of being drawn by a separate viewer.

const UNSUPPORTED_IN_USER_BLOCK = new Set(['derivative', 'setVariable', 'repeat']);

let userBlockEditorState = null;
let userBlockEditorToolbar = null;
let userBlockEditorTitle = null;

function isUserBlockWorkspaceEditing() {
  return Boolean(userBlockEditorState);
}

function deepCloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

// ---------- 묶기 (grouping) ----------

function startGroupSelection() {
  groupSelectionMode = true;
  groupSelectedIds.clear();
  cancelConnection();
  groupSelectBtn.hidden = true;
  createGroupBtn.hidden = false;
  cancelGroupBtn.hidden = false;
  updateGroupUI();
  connectionHint.textContent = '묶을 블록을 차례로 누르세요. 변수 블록은 자동으로 새 블록의 외부 입력으로 남습니다.';
}

function cancelGroupSelection() {
  groupSelectionMode = false;
  groupSelectedIds.clear();
  nodesLayer.querySelectorAll('.group-selected').forEach(el => el.classList.remove('group-selected'));
  groupSelectBtn.hidden = false;
  createGroupBtn.hidden = true;
  cancelGroupBtn.hidden = true;
  connectionHint.textContent = '블록을 놓고 수학식을 직접 만드세요.';
}

function toggleGroupNode(id) {
  // An external-input placeholder is an interface port of the block being
  // edited; grouping it would persist an editor-only block type inside another
  // definition.
  if (isUserBlockWorkspaceEditing() && graph.nodes.get(id)?.type === USER_BLOCK_EXTERNAL_INPUT_TYPE) {
    alert('외부 입력 블록은 사용자 블록의 입력 포트이므로 다른 블록과 다시 묶을 수 없습니다.');
    return;
  }

  if (groupSelectedIds.has(id)) groupSelectedIds.delete(id);
  else groupSelectedIds.add(id);
  nodesLayer.querySelector(`[data-node-id="${id}"]`)?.classList.toggle('group-selected', groupSelectedIds.has(id));
  selectedNodeId = id;
  renderInspector();
  updateGroupUI();
}

function updateGroupUI() {
  createGroupBtn.textContent = `새 블록 만들기 (${groupSelectedIds.size})`;
  createGroupBtn.disabled = groupSelectedIds.size === 0;
}

function createUserBlockFromSelection() {
  if (!groupSelectedIds.size) return;

  // 반복 and 값 바꾸기 are execution control, not pure functions, so they stay
  // outside a reusable block.
  for (const id of groupSelectedIds) {
    const node = graph.nodes.get(id);
    if (!node) continue;
    const special = getBlockDef(node.type).special;
    if (special === 'repeat' || special === 'setVariable') {
      alert('반복과 값 바꾸기는 실행 제어 블록이라 현재는 묶기 바깥에 두세요. 계산식 부분은 자유롭게 묶을 수 있습니다.');
      return;
    }
  }

  const variableIds = new Set([...groupSelectedIds].filter(id => graph.nodes.get(id)?.type === 'variable'));
  const core = new Set([...groupSelectedIds].filter(id => !variableIds.has(id)));
  if (!core.size) { alert('변수만으로는 새 블록을 만들 수 없습니다. 연산 블록도 함께 선택하세요.'); return; }
  for (const id of core) {
    if (getBlockDef(graph.nodes.get(id).type).special === 'derivative') {
      alert('미분 블록 자체는 아직 묶기 안에 넣지 마세요. 만든 블록 바깥에서 미분하면 됩니다.');
      return;
    }
  }

  const outgoingOutside = new Map();
  for (const id of core) outgoingOutside.set(id, graph.connections.filter(c => c.from === id && !core.has(c.to)));
  const outputCandidates = [...core].filter(id => {
    const hasInternalOut = graph.connections.some(c => c.from === id && core.has(c.to));
    const hasExternalOut = outgoingOutside.get(id).length > 0;
    return !hasInternalOut || hasExternalOut;
  });
  const unique = [...new Set(outputCandidates)];
  if (unique.length !== 1) { alert('새 블록은 출력이 하나여야 합니다. 선택한 블록들이 한 개의 마지막 출력으로 모이게 연결해 주세요.'); return; }
  const outputNodeId = unique[0];
  const name = prompt('새 블록 이름', '내 함수');
  if (!name?.trim()) return;

  const externalInputs = [];
  const internalConnections = [];
  for (const id of core) {
    const node = graph.nodes.get(id), def = getBlockDef(node.type);
    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const c = graph.connections.find(x => x.to === id && x.inputIndex === inputIndex);
      if (c && core.has(c.from)) internalConnections.push({ from: c.from, to: id, inputIndex });
      else externalInputs.push({ nodeId: id, inputIndex, label: def.inputs[inputIndex], fromNodeId: c?.from ?? null });
    }
  }

  const serializedNodes = [...core].map(id => {
    const n = graph.nodes.get(id);
    return { id: n.id, type: n.type, params: deepCloneJson(n.params) };
  });
  const customId = `u${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
  const definition = {
    id: customId,
    name: name.trim(),
    nodes: serializedNodes,
    connections: internalConnections,
    externalInputs: externalInputs.map(({ nodeId, inputIndex, label }) => ({ nodeId, inputIndex, label })),
    outputNodeId,
    formula: name.trim()
  };
  USER_BLOCKS.set(customId, definition);
  persistUserBlocks();
  renderMyBlocksPalette();

  const positions = [...core].map(id => graph.nodes.get(id));
  const x = positions.reduce((s, n) => s + n.x, 0) / positions.length;
  const y = positions.reduce((s, n) => s + n.y, 0) / positions.length;
  const outgoing = graph.connections.filter(c => c.from === outputNodeId && !core.has(c.to));
  removeNodes(core);
  const newNode = addBlock(`custom:${customId}`, { x, y });
  externalInputs.forEach((slot, inputIndex) => {
    if (slot.fromNodeId != null && graph.nodes.has(slot.fromNodeId)) {
      graph.connections.push({ from: slot.fromNodeId, to: newNode.id, inputIndex });
    }
  });
  outgoing.forEach(c => {
    if (graph.nodes.has(c.to)) {
      graph.connections = graph.connections.filter(x => !(x.to === c.to && x.inputIndex === c.inputIndex));
      graph.connections.push({ from: newNode.id, to: c.to, inputIndex: c.inputIndex });
    }
  });
  // Inside the 내부 구조 편집 workspace the grouped blocks are gone, so an output
  // pointer aimed at one of them would leave the definition unsaveable. The new
  // block produces exactly the value the old output did, so it takes over.
  if (userBlockEditorState && core.has(userBlockEditorState.outputNodeId)) {
    userBlockEditorState.outputNodeId = newNode.id;
  }
  cancelGroupSelection();
  updateWires();
  invalidatePreviews();
  syncWorkspaceState();
  decorateEditorNodes();
  notifyWorkspaceChanged();
}

// ---------- storage and palette ----------

// Every write to the library goes through here, so this is also where the
// compiled plans that captured the old definitions are marked stale.
function persistUserBlocks() {
  bumpUserBlockLibraryVersion();
  try {
    localStorage.setItem(USER_BLOCK_STORAGE_KEY, JSON.stringify([...USER_BLOCKS.values()]));
  } catch (e) {
    console.warn('사용자 블록 저장 실패', e);
  }
}

function loadUserBlocks() {
  try {
    const raw = localStorage.getItem(USER_BLOCK_STORAGE_KEY);
    if (!raw) return;
    for (const def of JSON.parse(raw)) USER_BLOCKS.set(def.id, def);
    bumpUserBlockLibraryVersion();
  } catch (e) {
    console.warn('사용자 블록 불러오기 실패', e);
  }
}

function renderMyBlocksPalette() {
  myBlocksPalette.replaceChildren();
  const definitions = [...USER_BLOCKS.values()];
  myBlocksEmpty.hidden = definitions.length > 0;

  for (const definition of definitions) {
    const row = document.createElement('div');
    row.className = 'my-block-palette-row';

    const addButton = document.createElement('button');
    addButton.className = 'palette-block custom my-block-add';
    addButton.dataset.customBlock = definition.id;
    addButton.textContent = definition.name;
    addButton.addEventListener('click', () => addBlock(`custom:${definition.id}`));

    const deleteButton = document.createElement('button');
    deleteButton.type = 'button';
    deleteButton.className = 'ghost danger my-block-delete';
    deleteButton.textContent = '삭제';
    deleteButton.title = `${definition.name} 사용자 블록 삭제`;
    deleteButton.addEventListener('click', event => {
      event.stopPropagation();
      deleteUserBlock(definition.id);
    });

    row.append(addButton, deleteButton);
    myBlocksPalette.appendChild(row);
  }
}

// Every 사용자 블록 id that `customId` uses, directly or through nested blocks.
function userBlockDependencies(customId, found = new Set()) {
  const definition = USER_BLOCKS.get(customId);
  if (!definition) return found;
  for (const node of definition.nodes || []) {
    const type = String(node.type || '');
    if (!type.startsWith('custom:')) continue;
    const childId = type.slice(7);
    if (found.has(childId)) continue;
    found.add(childId);
    userBlockDependencies(childId, found);
  }
  return found;
}

// Placing `candidateId` inside the definition of `hostId` closes a cycle when the
// candidate is the host itself or already depends on it. A saved cycle compiles
// without complaint and only shows up as a stack overflow on the next 계산, so
// it is refused while the block is being placed.
function userBlockWouldRecurse(candidateId, hostId) {
  if (!candidateId || !hostId) return false;
  if (candidateId === hostId) return true;
  return userBlockDependencies(candidateId).has(hostId);
}

function userBlockUsage(customId) {
  const type = `custom:${customId}`;
  const graphInstances = Array.from(graph.nodes.values()).filter(node => node.type === type).length;
  const parentBlocks = [];

  for (const definition of USER_BLOCKS.values()) {
    if (definition.id === customId) continue;
    if ((definition.nodes || []).some(node => node.type === type)) {
      parentBlocks.push(definition.name || definition.id);
    }
  }

  return { graphInstances, parentBlocks };
}

function deleteUserBlock(customId) {
  if (isUserBlockWorkspaceEditing()) {
    alert('사용자 블록 내부 편집을 저장하거나 취소한 뒤 블록을 삭제해 주세요.');
    return false;
  }

  const definition = USER_BLOCKS.get(customId);
  if (!definition) return false;

  const usage = userBlockUsage(customId);
  if (usage.graphInstances || usage.parentBlocks.length) {
    const reasons = [];
    if (usage.graphInstances) reasons.push(`현재 작업공간에서 ${usage.graphInstances}개 사용 중`);
    if (usage.parentBlocks.length) reasons.push(`다른 사용자 블록에서 사용 중: ${usage.parentBlocks.join(', ')}`);
    alert(`'${definition.name}' 블록을 아직 삭제할 수 없습니다.\n${reasons.join('\n')}\n먼저 해당 사용처를 제거해 주세요.`);
    return false;
  }

  if (!confirm(`사용자 블록 '${definition.name}'을 삭제할까요?\n저장된 내부 구조도 함께 삭제됩니다.`)) return false;

  USER_BLOCKS.delete(customId);
  persistUserBlocks();
  renderMyBlocksPalette();
  notifyWorkspaceChanged();
  return true;
}

// ---------- editing a definition in the workspace ----------

function ensureUserBlockEditorToolbar() {
  if (userBlockEditorToolbar) return;
  userBlockEditorToolbar = document.createElement('div');
  userBlockEditorToolbar.className = 'user-block-editor-toolbar';
  userBlockEditorToolbar.hidden = true;
  userBlockEditorToolbar.innerHTML = `
    <span class="pill user-block-editor-badge">사용자 블록 편집</span>
    <strong id="userBlockEditorTitle"></strong>
    <button type="button" id="userBlockEditorSave" class="primary">저장하고 돌아가기</button>
    <button type="button" id="userBlockEditorCancel" class="ghost">취소</button>`;
  document.querySelector('.workspace-head')?.appendChild(userBlockEditorToolbar);
  userBlockEditorTitle = document.getElementById('userBlockEditorTitle');
  document.getElementById('userBlockEditorSave')?.addEventListener('click', saveUserBlockEditor);
  document.getElementById('userBlockEditorCancel')?.addEventListener('click', cancelUserBlockEditor);
}

function setEditingControlsDisabled(disabled, state = userBlockEditorState) {
  const ids = [
    'evaluateBtn', 'evaluateSelectedBtn', 'resetWorkspaceBtn',
    'importProjectBtn', 'exportProjectBtn', 'importWeightsBtn', 'exportWeightsBtn',
    'classifyDrawingBtn'
  ];

  if (disabled) {
    state.disabledStates = new Map();
    for (const id of ids) {
      const element = document.getElementById(id);
      if (!element) continue;
      state.disabledStates.set(id, element.disabled);
      element.disabled = true;
    }
    return;
  }

  for (const [id, wasDisabled] of state?.disabledStates || []) {
    const element = document.getElementById(id);
    if (element) element.disabled = wasDisabled;
  }
}

// Older definitions were saved without positions; lay them out left to right by
// dependency depth so the internals are readable the first time they are opened.
function layoutDefinition(definition) {
  const nodes = (definition.nodes || []).map(node => deepCloneJson(node));
  if (nodes.every(node => Number.isFinite(Number(node.x)) && Number.isFinite(Number(node.y)))) {
    return nodes;
  }

  const nodeById = new Map(nodes.map(node => [Number(node.id), node]));
  const incoming = new Map(nodes.map(node => [Number(node.id), []]));
  for (const connection of definition.connections || []) {
    if (incoming.has(Number(connection.to)) && nodeById.has(Number(connection.from))) {
      incoming.get(Number(connection.to)).push(Number(connection.from));
    }
  }

  const layerMemo = new Map();
  const visiting = new Set();
  const layerOf = id => {
    if (layerMemo.has(id)) return layerMemo.get(id);
    if (visiting.has(id)) return 0;
    visiting.add(id);
    let layer = 0;
    for (const parent of incoming.get(id) || []) layer = Math.max(layer, layerOf(parent) + 1);
    visiting.delete(id);
    layerMemo.set(id, layer);
    return layer;
  };

  const layers = new Map();
  for (const node of nodes) {
    const layer = layerOf(Number(node.id));
    if (!layers.has(layer)) layers.set(layer, []);
    layers.get(layer).push(node);
  }

  const startX = 260;
  const startY = 60;
  const gapX = 245;
  const gapY = 145;
  for (const [layer, layerNodes] of layers) {
    layerNodes.forEach((node, row) => {
      node.x = startX + layer * gapX;
      node.y = startY + row * gapY;
    });
  }
  return nodes;
}

function buildEditorGraph(definition) {
  const internalNodes = layoutDefinition(definition);
  const nodes = new Map();
  let largestId = 0;

  for (const saved of internalNodes) {
    const id = Number(saved.id);
    if (!Number.isInteger(id) || id <= 0 || nodes.has(id)) {
      throw new Error('사용자 블록 내부 노드 ID가 올바르지 않습니다.');
    }
    getBlockDef(String(saved.type));
    const node = {
      id,
      type: String(saved.type),
      x: Number.isFinite(Number(saved.x)) ? Number(saved.x) : 260,
      y: Number.isFinite(Number(saved.y)) ? Number(saved.y) : 60,
      params: saved.params && typeof saved.params === 'object' ? deepCloneJson(saved.params) : {}
    };
    nodes.set(id, node);
    largestId = Math.max(largestId, id);
  }

  const connections = (definition.connections || []).map(connection => ({
    from: Number(connection.from),
    to: Number(connection.to),
    inputIndex: Number(connection.inputIndex)
  })).filter(connection => nodes.has(connection.from) && nodes.has(connection.to));

  // External inputs become negative-id placeholder nodes on the left edge.
  const externalInputs = Array.isArray(definition.externalInputs) ? definition.externalInputs : [];
  externalInputs.forEach((slot, index) => {
    const id = -(index + 1);
    nodes.set(id, {
      id,
      type: USER_BLOCK_EXTERNAL_INPUT_TYPE,
      x: 35,
      y: 60 + index * 135,
      params: {
        label: String(slot.label || `입력 ${index + 1}`),
        externalIndex: index
      }
    });
    const targetId = Number(slot.nodeId);
    const inputIndex = Number(slot.inputIndex);
    if (nodes.has(targetId) && Number.isInteger(inputIndex) && inputIndex >= 0) {
      connections.push({ from: id, to: targetId, inputIndex });
    }
  });

  return { nodes, connections, nextId: largestId + 1 };
}

function decorateEditorNodes() {
  if (!userBlockEditorState) return;
  nodesLayer.querySelectorAll('.node').forEach(element => {
    const id = Number(element.dataset.nodeId);
    const node = graph.nodes.get(id);
    element.classList.toggle('user-block-editor-external', node?.type === USER_BLOCK_EXTERNAL_INPUT_TYPE);
    element.classList.toggle('user-block-editor-output', id === userBlockEditorState.outputNodeId);
  });
}

function redrawEditorWorkspace() {
  renderWorkspaceGraph();
  pendingOutput = null;
  selectedConnection = null;
  selectedNodeId = null;
  renderInspector();
  syncWorkspaceState();
  updateWires();
  decorateEditorNodes();
}

function openUserBlockEditor(definition) {
  if (!definition || userBlockEditorState) return;
  if (evaluateBtn.disabled) {
    alert('계산이 끝난 뒤 사용자 블록을 편집해 주세요.');
    return;
  }

  cancelGroupSelection();
  cancelConnection();
  const built = buildEditorGraph(definition);

  userBlockEditorState = {
    customId: definition.id,
    outputNodeId: Number(definition.outputNodeId),
    outerNodes: graph.nodes,
    outerConnections: graph.connections,
    outerNextNodeId: nextNodeId,
    outerSelectedNodeId: selectedNodeId,
    outerViewport: getWorkspaceViewSnapshot(),
    userBlocksSnapshot: deepCloneJson([...USER_BLOCKS.values()]),
    disabledStates: new Map()
  };

  graph.nodes = built.nodes;
  graph.connections = built.connections;
  nextNodeId = built.nextId;

  ensureUserBlockEditorToolbar();
  userBlockEditorTitle.textContent = definition.name || '사용자 블록';
  userBlockEditorToolbar.hidden = false;
  setEditingControlsDisabled(true);
  redrawEditorWorkspace();

  restoreWorkspaceViewSnapshot(definition.editorViewport || { x: 0, y: 0, zoom: 1 });
  workspaceStatus.textContent = `${definition.nodes?.length || 0}개 내부 블록`;
}

function collectEditedDefinition() {
  if (!userBlockEditorState) throw new Error('편집 중인 사용자 블록이 없습니다.');
  const current = USER_BLOCKS.get(userBlockEditorState.customId);
  if (!current) throw new Error('편집 중인 사용자 블록 정의를 찾을 수 없습니다.');

  const internalNodes = [...graph.nodes.values()].filter(node => node.type !== USER_BLOCK_EXTERNAL_INPUT_TYPE);
  const internalIds = new Set(internalNodes.map(node => node.id));
  if (!internalIds.has(userBlockEditorState.outputNodeId)) {
    throw new Error('출력 블록이 없습니다. 내부 블록 하나를 선택해 출력으로 지정하세요.');
  }

  const externalNodes = [...graph.nodes.values()]
    .filter(node => node.type === USER_BLOCK_EXTERNAL_INPUT_TYPE)
    .sort((a, b) => Number(a.params.externalIndex) - Number(b.params.externalIndex));
  const externalIds = new Set(externalNodes.map(node => node.id));

  const internalConnections = [];
  for (const connection of graph.connections) {
    if (internalIds.has(connection.from) && internalIds.has(connection.to)) {
      internalConnections.push({ ...connection });
    }
  }

  const externalInputs = [];
  for (const externalNode of externalNodes) {
    const outgoing = graph.connections.filter(connection => connection.from === externalNode.id && internalIds.has(connection.to));
    if (outgoing.length !== 1) {
      throw new Error(`외부 입력 '${externalNode.params.label || '입력'}'은 내부 입력 한 곳에 정확히 연결되어야 합니다.`);
    }
    const connection = outgoing[0];
    const target = graph.nodes.get(connection.to);
    const targetDef = getBlockDef(target.type);
    if (!Number.isInteger(connection.inputIndex) || connection.inputIndex < 0 || connection.inputIndex >= targetDef.inputs.length) {
      throw new Error('외부 입력 연결 위치가 올바르지 않습니다.');
    }
    externalInputs.push({
      nodeId: connection.to,
      inputIndex: connection.inputIndex,
      label: String(externalNode.params.label || targetDef.inputs[connection.inputIndex] || '입력')
    });
  }

  // Every internal input must be satisfied by either another internal node or
  // one of the explicit external-input source nodes.
  for (const node of internalNodes) {
    const def = getBlockDef(node.type);
    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const matches = graph.connections.filter(connection => connection.to === node.id && connection.inputIndex === inputIndex);
      if (matches.length !== 1) {
        throw new Error(`'${def.title}' 블록의 '${def.inputs[inputIndex]}' 입력을 정확히 하나 연결해 주세요.`);
      }
      const sourceId = matches[0].from;
      if (!internalIds.has(sourceId) && !externalIds.has(sourceId)) {
        throw new Error('사용자 블록 내부에 올바르지 않은 연결이 있습니다.');
      }
    }
  }

  const nodes = internalNodes.map(node => ({
    id: node.id,
    type: node.type,
    x: node.x,
    y: node.y,
    params: deepCloneJson(node.params || {})
  }));

  return {
    ...deepCloneJson(current),
    nodes,
    connections: internalConnections,
    externalInputs,
    outputNodeId: userBlockEditorState.outputNodeId,
    editorViewport: getWorkspaceViewSnapshot()
  };
}

function restoreOuterWorkspace(finishedState) {
  graph.nodes = finishedState.outerNodes;
  graph.connections = finishedState.outerConnections;
  nextNodeId = finishedState.outerNextNodeId;
  selectedNodeId = finishedState.outerSelectedNodeId;
  pendingOutput = null;
  selectedConnection = null;

  renderWorkspaceGraph();
  setEditingControlsDisabled(false, finishedState);
  if (userBlockEditorToolbar) userBlockEditorToolbar.hidden = true;
  syncWorkspaceState();
  updateWires();
  restoreWorkspaceViewSnapshot(finishedState.outerViewport);
  renderInspector();
  renderMyBlocksPalette();
}

function saveUserBlockEditor() {
  if (!userBlockEditorState) return;

  // Protect imported or legacy definitions that somehow contain a block the
  // user-block runtime cannot execute as a pure function.
  const forbidden = [...graph.nodes.values()].find(node => UNSUPPORTED_IN_USER_BLOCK.has(node.type));
  if (forbidden) {
    let title = forbidden.type;
    try { title = getBlockDef(forbidden.type).title; } catch { /* keep the raw type */ }
    alert(`'${title}' 블록은 사용자 블록 내부에 저장할 수 없습니다. 제거한 뒤 다시 저장해 주세요.`);
    return;
  }

  // A block that contains itself, directly or through a nested block, saves and
  // persists happily and then blows the stack on the next 계산.
  const recursive = [...graph.nodes.values()].find(node =>
    String(node.type).startsWith('custom:')
    && userBlockWouldRecurse(String(node.type).slice(7), userBlockEditorState.customId));
  if (recursive) {
    const recursiveId = String(recursive.type).slice(7);
    const host = USER_BLOCKS.get(userBlockEditorState.customId);
    const hostName = host?.name || '이 블록';
    alert(recursiveId === userBlockEditorState.customId
      ? `'${hostName}' 자신이 내부에 들어 있습니다. 제거한 뒤 다시 저장해 주세요.`
      : `'${USER_BLOCKS.get(recursiveId)?.name || recursiveId}' 블록은 '${hostName}'을 사용하고 있어 내부에 둘 수 없습니다. 제거한 뒤 다시 저장해 주세요.`);
    return;
  }

  let updated;
  try {
    updated = collectEditedDefinition();
  } catch (error) {
    alert(`사용자 블록을 저장할 수 없습니다.\n${error.message}`);
    return;
  }

  const finishedState = userBlockEditorState;
  // A new object invalidates the compiled-plan WeakMap cache.
  USER_BLOCKS.set(finishedState.customId, updated);
  persistUserBlocks();
  userBlockEditorState = null;
  restoreOuterWorkspace(finishedState);
  // Announced only after the outer graph is back, so the autosave snapshot holds
  // the normal workspace plus the newly edited definition.
  notifyWorkspaceChanged();
}

function cancelUserBlockEditor() {
  if (!userBlockEditorState) return;
  if (!confirm('사용자 블록 편집을 취소할까요?\n내부에서 바꾼 내용은 저장되지 않습니다.')) return;

  const finishedState = userBlockEditorState;
  USER_BLOCKS.clear();
  for (const definition of finishedState.userBlocksSnapshot) USER_BLOCKS.set(definition.id, definition);
  persistUserBlocks();
  userBlockEditorState = null;
  restoreOuterWorkspace(finishedState);
}

function setSelectedAsUserBlockOutput() {
  if (!userBlockEditorState || selectedNodeId == null) return;
  const node = graph.nodes.get(selectedNodeId);
  if (!node || node.type === USER_BLOCK_EXTERNAL_INPUT_TYPE) return;
  userBlockEditorState.outputNodeId = node.id;
  decorateEditorNodes();
  renderInspector();
}

// Inspector additions: open a custom block, or pick the output while editing.
INSPECTOR_EXTENSIONS.push(node => {
  if (inspectorContent.hidden) return;

  if (userBlockEditorState) {
    if (node.type === USER_BLOCK_EXTERNAL_INPUT_TYPE) {
      const note = document.createElement('p');
      note.className = 'muted user-block-editor-input-note';
      note.textContent = '이 블록은 접힌 사용자 블록의 외부 입력 포트입니다.';
      inspectorControls.appendChild(note);
      return;
    }

    const outputButton = document.createElement('button');
    outputButton.type = 'button';
    outputButton.className = 'ghost user-block-output-button';
    outputButton.textContent = node.id === userBlockEditorState.outputNodeId ? '현재 출력 블록' : '이 블록을 출력으로 지정';
    outputButton.disabled = node.id === userBlockEditorState.outputNodeId;
    outputButton.addEventListener('click', setSelectedAsUserBlockOutput);
    inspectorControls.prepend(outputButton);
    return;
  }

  if (!node.type?.startsWith('custom:')) return;
  const definition = USER_BLOCKS.get(node.type.slice(7));
  if (!definition) return;

  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'primary user-block-open-button';
  button.textContent = '내부 구조 편집';
  button.addEventListener('click', () => openUserBlockEditor(definition));
  inspectorControls.prepend(button);
});

// ---------- editor guards ----------

// External-input placeholders are part of the custom block's public interface;
// deleting one directly would silently change the number of outer ports.
deleteNodeBtn.addEventListener('click', event => {
  if (!userBlockEditorState) return;
  const node = selectedNodeId == null ? null : graph.nodes.get(selectedNodeId);
  if (node?.type !== USER_BLOCK_EXTERNAL_INPUT_TYPE) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  alert('외부 입력 블록은 직접 삭제할 수 없습니다. 연결 대상과 이름은 편집할 수 있습니다.');
}, true);

// Never let the global workspace reset destroy the temporarily swapped graph.
resetWorkspaceBtn.addEventListener('click', event => {
  if (!userBlockEditorState) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  alert('사용자 블록 편집을 저장하거나 취소한 뒤 작업공간을 초기화해 주세요.');
}, true);

// Unsupported stateful/special blocks must not reach the palette's addBlock
// handler while a definition is open.
document.addEventListener('click', event => {
  if (!userBlockEditorState) return;
  const button = event.target.closest('[data-block]');
  if (!button) return;
  const type = String(button.dataset.block || '');

  // 묶기 never puts a 변수 inside a definition — it keeps it outside and leaves an
  // external input in its place — because a 사용자 블록 is a pure function of its
  // ports. A 변수 dropped straight into the internals would be invisible from the
  // outside and could never receive a gradient.
  if (type === 'variable') {
    event.preventDefault();
    event.stopImmediatePropagation();
    alert('변수 블록은 사용자 블록 내부에 넣을 수 없습니다. 바깥에 두고 외부 입력으로 연결해 주세요.');
    return;
  }

  if (!UNSUPPORTED_IN_USER_BLOCK.has(type)) return;

  event.preventDefault();
  event.stopImmediatePropagation();
  alert('미분, 값 바꾸기, 반복 블록은 사용자 블록 내부에 넣을 수 없습니다. 사용자 블록 바깥에서 사용해 주세요.');
}, true);

// The palette still lists the block currently being edited, and the blocks that
// use it. Adding one of those closes a cycle, so stop it at the palette instead
// of letting it become a stack overflow at 계산 time.
document.addEventListener('click', event => {
  if (!userBlockEditorState) return;
  const button = event.target.closest('[data-custom-block]');
  if (!button) return;
  const candidateId = String(button.dataset.customBlock || '');
  if (!userBlockWouldRecurse(candidateId, userBlockEditorState.customId)) return;

  event.preventDefault();
  event.stopImmediatePropagation();
  const host = USER_BLOCKS.get(userBlockEditorState.customId);
  alert(candidateId === userBlockEditorState.customId
    ? `'${host?.name || '이 블록'}'을 자기 자신 안에 넣을 수는 없습니다.`
    : `'${USER_BLOCKS.get(candidateId)?.name || candidateId}' 블록은 '${host?.name || '이 블록'}'을 사용하고 있어 내부에 넣을 수 없습니다.`);
}, true);

const userBlockPaletteStyle = document.createElement('style');
userBlockPaletteStyle.textContent = `
  .my-block-palette-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 5px; align-items: stretch; margin: 4px 0; }
  .my-block-palette-row .palette-block { margin: 0; min-width: 0; }
  .my-block-delete { padding: 7px 8px; font-size: 11px; }
`;
document.head.appendChild(userBlockPaletteStyle);

// Kept for compatibility with anything reading the old global.
window.isUserBlockWorkspaceEditing = isUserBlockWorkspaceEditing;
