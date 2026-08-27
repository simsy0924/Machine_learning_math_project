// Rendering and direct manipulation of the workspace: nodes, wires, the
// inspector and the value previews. Panning, zooming and node dragging live in
// js/viewport.js, which owns the pointer gestures.

function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }

function escapeHtml(text) {
  return String(text).replace(/[&<>'"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[ch]);
}

function kindLabel(kind) {
  return ({ source: '입력', data: '데이터', generator: '생성', transform: '변환', operation: '연산', calculus: '미분', custom: '내 블록', sink: '확인' })[kind] || kind;
}

function formatValue(value, compact = false) {
  if (value === undefined) return '계산 전';
  if (typeof value === 'number') return Number.isFinite(value) ? String(Number(value.toFixed(6))) : String(value);
  if (value?.kind === 'image') return '그림';
  if (isArrayValue(value)) {
    const shape = value.shape.join('×');
    if (compact) return `배열 ${shape}`;
    const sample = Array.from(value.data.slice(0, 18)).map(v => Number(v.toFixed(3)));
    return `shape: [${value.shape.join(', ')}]\n[${sample.join(', ')}${value.data.length > sample.length ? ', …' : ''}]`;
  }
  if (value?.kind === 'dataset') return `데이터 ${value.classes.length}종`;
  if (value?.kind === 'datasetClass') return `종류: ${value.name}`;
  if (value?.kind === 'sample') return `${value.className} #${value.index}`;
  return String(value);
}

function updateNodePreview(node) {
  const el = nodesLayer.querySelector(`[data-node-id="${node.id}"] .node-preview`);
  if (el) el.textContent = node.lastError ? `오류: ${node.lastError}` : formatValue(node.lastValue, true);
}

function invalidatePreviews() {
  for (const node of graph.nodes.values()) {
    node.lastValue = undefined;
    node.lastError = null;
    updateNodePreview(node);
  }
  renderInspector();
}

// ---------- nodes ----------

// Which input ports are currently drawn as connected. Kept so the marking pass
// below can skip the DOM while nothing about the wiring has changed.
let markedInputPortKeys = new Set();

function renderNode(node) {
  const def = getBlockDef(node.type);
  const el = document.createElement('div');
  el.className = `node ${def.kind}`;
  el.dataset.nodeId = node.id;
  el.style.left = `${node.x}px`;
  el.style.top = `${node.y}px`;
  const inputsHtml = def.inputs.map((label, index) => {
    const safeLabel = escapeHtml(label);
    return `<div class="port-row"><span class="port input" data-node-id="${node.id}" data-port-index="${index}" data-port-label="${safeLabel}" title="입력 ${safeLabel}"></span><span class="port-label">${safeLabel}</span></div>`;
  }).join('');
  el.innerHTML = `<div class="node-head"><span>${escapeHtml(def.title)}</span><span class="node-kind">${kindLabel(def.kind)}</span></div><div class="node-body">${inputsHtml}${def.inputs.length ? '' : '<span class="muted">입력 없음</span>'}<div class="node-preview">계산 전</div><div class="port-row output-row"><span class="port-label">출력</span><span class="port output" data-node-id="${node.id}" title="출력"></span></div></div>`;
  nodesLayer.appendChild(el);
  markedInputPortKeys.clear(); // these ports are new, so they carry no marks yet

  el.querySelector('.port.output').addEventListener('click', event => {
    event.stopPropagation();
    if (!groupSelectionMode) beginConnection(node.id, event.currentTarget);
  });
  el.querySelectorAll('.port.input').forEach(port => port.addEventListener('click', event => {
    event.stopPropagation();
    if (!groupSelectionMode) clickInputPort(node.id, Number(port.dataset.portIndex));
  }));
}

// Draw every node of graph.* from scratch. Used after an import and when the
// 사용자 블록 editor swaps the visible graph.
function renderWorkspaceGraph() {
  nodesLayer.replaceChildren();
  wiresSvg.replaceChildren();
  for (const node of graph.nodes.values()) {
    renderNode(node);
    updateNodePreview(node);
  }
}

function addBlock(type, position = null) {
  const def = getBlockDef(type);
  const id = nextNodeId++;
  const count = graph.nodes.size;
  const node = { id, type, x: position?.x ?? 34 + (count % 4) * 205, y: position?.y ?? 34 + Math.floor(count / 4) * 135, params: {} };
  for (const control of def.controls || []) node.params[control.key] = control.default;
  graph.nodes.set(id, node);
  renderNode(node);
  selectNode(id);
  syncWorkspaceState();
  updateWires();
  notifyWorkspaceChanged();
  return node;
}

function removeNodes(ids) {
  for (const id of ids) {
    graph.nodes.delete(id);
    nodesLayer.querySelector(`[data-node-id="${id}"]`)?.remove();
  }
  graph.connections = graph.connections.filter(c => !ids.has(c.from) && !ids.has(c.to));
  if (selectedConnection && (ids.has(selectedConnection.to) || !findConnection(selectedConnection.to, selectedConnection.inputIndex))) {
    selectedConnection = null;
  }
  notifyWorkspaceChanged();
}

function deleteSelectedNode() {
  if (selectedNodeId == null) return;
  removeNodes(new Set([selectedNodeId]));
  selectedNodeId = null;
  renderInspector();
  syncWorkspaceState();
  updateWires();
}

function selectNode(id) {
  selectedNodeId = id;
  nodesLayer.querySelectorAll('.node').forEach(el => el.classList.toggle('selected', Number(el.dataset.nodeId) === id));
  renderInspector();
}

// ---------- connections ----------

const DEFAULT_CONNECTION_HINT = '블록을 놓고 수학식을 직접 만드세요.';

function setConnectionHint(text) {
  if (!groupSelectionMode) connectionHint.textContent = text;
}

function beginConnection(nodeId, portEl) {
  cancelConnection();
  pendingOutput = nodeId;
  portEl.classList.add('hot');
  document.querySelectorAll('.port.input').forEach(p => p.classList.add('hot'));
  setConnectionHint('연결할 블록의 왼쪽 입력 ○를 누르세요.');
}

// One input port holds at most one wire, so (to, inputIndex) identifies it.
function findConnection(toNodeId, inputIndex) {
  return graph.connections.find(c => c.to === toNodeId && c.inputIndex === inputIndex) || null;
}

function isSelectedConnection(connection) {
  return Boolean(selectedConnection)
    && selectedConnection.to === connection.to
    && selectedConnection.inputIndex === connection.inputIndex;
}

// Clicking an input port finishes the wire being drawn, or — when no wire is in
// flight — cuts the one that port already holds.
function clickInputPort(toNodeId, inputIndex) {
  if (pendingOutput != null) finishConnection(toNodeId, inputIndex);
  else disconnectInput(toNodeId, inputIndex);
}

function finishConnection(toNodeId, inputIndex) {
  if (pendingOutput == null) return;
  if (pendingOutput === toNodeId) return cancelConnection();
  graph.connections = graph.connections.filter(c => !(c.to === toNodeId && c.inputIndex === inputIndex));
  graph.connections.push({ from: pendingOutput, to: toNodeId, inputIndex });
  cancelConnection();
  updateWires();
  invalidatePreviews();
  notifyWorkspaceChanged();
}

// Cut the wire feeding one input. Returns whether anything was connected.
function disconnectInput(toNodeId, inputIndex) {
  if (!findConnection(toNodeId, inputIndex)) return false;
  graph.connections = graph.connections.filter(c => !(c.to === toNodeId && c.inputIndex === inputIndex));
  selectedConnection = null;
  updateWires();
  invalidatePreviews();
  notifyWorkspaceChanged();
  setConnectionHint('연결선을 끊었습니다. 오른쪽 ●부터 다시 연결할 수 있습니다.');
  return true;
}

function disconnectSelectedConnection() {
  if (!selectedConnection) return false;
  return disconnectInput(selectedConnection.to, selectedConnection.inputIndex);
}

// Picking a wire is deliberately a two-step gesture: the first click marks it so
// the user can see which one is about to go, the second cuts it.
function selectConnection(toNodeId, inputIndex) {
  const connection = findConnection(toNodeId, inputIndex);
  if (!connection) return;
  cancelConnection();
  selectedConnection = { to: connection.to, inputIndex: connection.inputIndex };
  updateWires();
  renderInspector();
  setConnectionHint('연결선을 선택했습니다. 한 번 더 누르거나 Delete 키를 누르면 끊어집니다.');
}

function clickWire(toNodeId, inputIndex) {
  if (isSelectedConnection({ to: toNodeId, inputIndex })) disconnectInput(toNodeId, inputIndex);
  else selectConnection(toNodeId, inputIndex);
}

function clearConnectionSelection() {
  if (!selectedConnection) return;
  selectedConnection = null;
  updateWires();
  renderInspector();
}

function cancelConnection() {
  pendingOutput = null;
  clearHotPorts();
  clearConnectionSelection();
  setConnectionHint(DEFAULT_CONNECTION_HINT);
}

function clearHotPorts() {
  document.querySelectorAll('.port.hot').forEach(el => el.classList.remove('hot'));
}

function wirePath(className, geometry, connection) {
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('class', className);
  path.setAttribute('d', geometry);
  path.dataset.to = connection.to;
  path.dataset.inputIndex = connection.inputIndex;
  return path;
}

function updateWires() {
  wiresSvg.setAttribute('viewBox', `0 0 ${workspace.clientWidth} ${workspace.clientHeight}`);
  wiresSvg.replaceChildren();
  for (const c of graph.connections) {
    const fromEl = nodesLayer.querySelector(`[data-node-id="${c.from}"] .port.output`);
    const toEl = nodesLayer.querySelector(`[data-node-id="${c.to}"] .port.input[data-port-index="${c.inputIndex}"]`);
    if (!fromEl || !toEl) continue;
    const a = portCenter(fromEl), b = portCenter(toEl);
    const bend = Math.max(60, Math.abs(b.x - a.x) * 0.45);
    const geometry = `M ${a.x} ${a.y} C ${a.x + bend} ${a.y}, ${b.x - bend} ${b.y}, ${b.x} ${b.y}`;
    // A transparent, much thicker copy of the wire sits under the visible one:
    // a 3px stroke is far too thin to hit with a finger or a quick click.
    wiresSvg.append(wirePath('wire-hit', geometry, c), wirePath(isSelectedConnection(c) ? 'wire selected' : 'wire', geometry, c));
  }
  markConnectedInputPorts();
}

// A filled input port is one that already holds a wire, and so one that can be
// clicked to disconnect. updateWires runs on every frame of a node drag, so the
// ports are only walked when the set of connected inputs actually changed.
function markConnectedInputPorts() {
  const connected = new Set();
  for (const c of graph.connections) connected.add(`${c.to}:${c.inputIndex}`);

  let unchanged = connected.size === markedInputPortKeys.size;
  if (unchanged) {
    for (const key of connected) {
      if (!markedInputPortKeys.has(key)) {
        unchanged = false;
        break;
      }
    }
  }
  if (unchanged) return;

  for (const port of nodesLayer.querySelectorAll('.port.input')) {
    const isConnected = connected.has(`${port.dataset.nodeId}:${port.dataset.portIndex}`);
    const label = port.dataset.portLabel || '';
    port.classList.toggle('connected', isConnected);
    port.title = isConnected ? `입력 ${label} — 누르면 연결이 끊어집니다` : `입력 ${label}`;
  }
  markedInputPortKeys = connected;
}

// ---------- inspector ----------

function renderInspector() {
  const node = selectedNodeId == null ? null : graph.nodes.get(selectedNodeId);
  inspectorEmpty.hidden = Boolean(node);
  inspectorContent.hidden = !node;
  deleteNodeBtn.disabled = !node;
  if (!node) {
    inspectorConnections.replaceChildren();
    return;
  }

  const def = getBlockDef(node.type);
  inspectorTitle.textContent = def.title;
  inspectorDescription.textContent = def.description;
  inspectorFormula.textContent = def.formula(node);
  inspectorValue.textContent = node.lastError ? `오류: ${node.lastError}` : formatValue(node.lastValue);
  inspectorControls.replaceChildren();

  for (const control of def.controls || []) {
    const row = document.createElement('div');
    row.className = 'control-row';
    const label = document.createElement('label');
    label.textContent = control.label;

    let input;
    if (control.type === 'select') {
      input = document.createElement('select');
      for (const option of control.options || []) {
        const el = document.createElement('option');
        el.value = option.value;
        el.textContent = option.label;
        input.appendChild(el);
      }
      input.value = node.params[control.key];
    } else {
      input = document.createElement('input');
      input.type = control.type || 'text';
      if (control.step) input.step = control.step;
      input.value = node.params[control.key];
    }

    input.addEventListener('input', () => {
      node.params[control.key] = input.value;
      inspectorFormula.textContent = def.formula(node);
      for (const graphNode of graph.nodes.values()) {
        graphNode.lastValue = undefined;
        graphNode.lastError = null;
        updateNodePreview(graphNode);
      }
      inspectorValue.textContent = '계산 전';
    });

    row.append(label, input);
    inspectorControls.appendChild(row);
  }

  renderInspectorConnections(node, def);
  for (const extend of INSPECTOR_EXTENSIONS) extend(node, def);
}

// The wires reaching this block, each with a button that cuts it. Clicking a
// thin wire is awkward on a touch screen, so the inspector offers the same cut
// as a plain list.
function renderInspectorConnections(node, def) {
  inspectorConnections.replaceChildren();

  const connected = [];
  for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
    const connection = findConnection(node.id, inputIndex);
    if (connection) connected.push({ inputIndex, connection });
  }
  if (!connected.length) return;

  const heading = document.createElement('span');
  heading.className = 'connection-list-title';
  heading.textContent = '연결된 입력';
  inspectorConnections.appendChild(heading);

  for (const { inputIndex, connection } of connected) {
    const source = graph.nodes.get(connection.from);
    const row = document.createElement('div');
    row.className = 'connection-row';
    if (isSelectedConnection(connection)) row.classList.add('selected');

    const label = document.createElement('span');
    label.className = 'connection-row-label';
    label.textContent = `${def.inputs[inputIndex]} ← ${source ? getBlockDef(source.type).title : '알 수 없는 블록'}`;

    const cut = document.createElement('button');
    cut.type = 'button';
    cut.className = 'ghost danger connection-cut';
    cut.textContent = '연결 끊기';
    cut.addEventListener('click', () => disconnectInput(node.id, inputIndex));

    row.append(label, cut);
    inspectorConnections.appendChild(row);
  }
}

// ---------- workspace-wide state ----------

function syncWorkspaceState() {
  emptyState.hidden = graph.nodes.size > 0;
  workspaceStatus.textContent = `${graph.nodes.size}개 블록`;
}

function resetWorkspace() {
  // Resetting the workspace also drops trained weights and any half-applied
  // 반복 state; nothing in the old graph should survive into the new one.
  RUNTIME_VARIABLES.clear();
  pendingVariableUpdates = null;

  graph.nodes.clear();
  graph.connections = [];
  nodesLayer.replaceChildren();
  wiresSvg.replaceChildren();
  nextNodeId = 1;
  selectedNodeId = null;
  cancelGroupSelection();
  cancelConnection();
  renderInspector();
  syncWorkspaceState();
  resetWorkspaceViewState();
  notifyWorkspaceChanged();
}
