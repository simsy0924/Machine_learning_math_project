function startGroupSelection() {
  groupSelectionMode = true; groupSelectedIds.clear(); cancelConnection(); groupSelectBtn.hidden = true; createGroupBtn.hidden = false; cancelGroupBtn.hidden = false; updateGroupUI(); connectionHint.textContent = '묶을 블록을 차례로 누르세요. 변수 블록은 자동으로 새 블록의 외부 입력으로 남습니다.';
}
function cancelGroupSelection() { groupSelectionMode = false; groupSelectedIds.clear(); nodesLayer.querySelectorAll('.group-selected').forEach(el => el.classList.remove('group-selected')); groupSelectBtn.hidden = false; createGroupBtn.hidden = true; cancelGroupBtn.hidden = true; connectionHint.textContent = '블록을 놓고 수학식을 직접 만드세요.'; }
function toggleGroupNode(id) { if (groupSelectedIds.has(id)) groupSelectedIds.delete(id); else groupSelectedIds.add(id); const el = nodesLayer.querySelector(`[data-node-id="${id}"]`); el?.classList.toggle('group-selected', groupSelectedIds.has(id)); selectedNodeId = id; renderInspector(); updateGroupUI(); }
function updateGroupUI() { createGroupBtn.textContent = `새 블록 만들기 (${groupSelectedIds.size})`; createGroupBtn.disabled = groupSelectedIds.size === 0; }

function createUserBlockFromSelection() {
  if (!groupSelectedIds.size) return;
  const variableIds = new Set([...groupSelectedIds].filter(id => graph.nodes.get(id)?.type === 'variable'));
  const core = new Set([...groupSelectedIds].filter(id => !variableIds.has(id)));
  if (!core.size) { alert('변수만으로는 새 블록을 만들 수 없습니다. 연산 블록도 함께 선택하세요.'); return; }
  for (const id of core) if (getBlockDef(graph.nodes.get(id).type).special === 'derivative') { alert('미분 블록 자체는 아직 묶기 안에 넣지 마세요. 만든 블록 바깥에서 미분하면 됩니다.'); return; }

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
  const name = prompt('새 블록 이름', '내 함수'); if (!name?.trim()) return;

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
  const serializedNodes = [...core].map(id => { const n = graph.nodes.get(id); return { id: n.id, type: n.type, params: JSON.parse(JSON.stringify(n.params)) }; });
  const customId = `u${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
  const definition = { id: customId, name: name.trim(), nodes: serializedNodes, connections: internalConnections, externalInputs: externalInputs.map(({ nodeId, inputIndex, label }) => ({ nodeId, inputIndex, label })), outputNodeId, formula: name.trim() };
  USER_BLOCKS.set(customId, definition); persistUserBlocks(); renderMyBlocksPalette();

  const positions = [...core].map(id => graph.nodes.get(id)); const x = positions.reduce((s, n) => s + n.x, 0) / positions.length; const y = positions.reduce((s, n) => s + n.y, 0) / positions.length;
  const outgoing = graph.connections.filter(c => c.from === outputNodeId && !core.has(c.to));
  removeNodes(core);
  const newNode = addBlock(`custom:${customId}`, { x, y });
  externalInputs.forEach((slot, inputIndex) => { if (slot.fromNodeId != null && graph.nodes.has(slot.fromNodeId)) graph.connections.push({ from: slot.fromNodeId, to: newNode.id, inputIndex }); });
  outgoing.forEach(c => { if (graph.nodes.has(c.to)) { graph.connections = graph.connections.filter(x => !(x.to === c.to && x.inputIndex === c.inputIndex)); graph.connections.push({ from: newNode.id, to: c.to, inputIndex: c.inputIndex }); } });
  cancelGroupSelection(); updateWires(); invalidatePreviews(); syncWorkspaceState();
}

function persistUserBlocks() { try { localStorage.setItem(USER_BLOCK_STORAGE_KEY, JSON.stringify([...USER_BLOCKS.values()])); } catch (e) { console.warn('사용자 블록 저장 실패', e); } }
function loadUserBlocks() { try { const raw = localStorage.getItem(USER_BLOCK_STORAGE_KEY); if (!raw) return; for (const def of JSON.parse(raw)) USER_BLOCKS.set(def.id, def); } catch (e) { console.warn('사용자 블록 불러오기 실패', e); } }
function renderMyBlocksPalette() {
  myBlocksPalette.replaceChildren(); const defs = [...USER_BLOCKS.values()]; myBlocksEmpty.hidden = defs.length > 0;
  for (const def of defs) { const btn = document.createElement('button'); btn.className = 'palette-block custom'; btn.dataset.customBlock = def.id; btn.textContent = def.name; btn.addEventListener('click', () => addBlock(`custom:${def.id}`)); myBlocksPalette.appendChild(btn); }
}

function updateNodePreview(node) { const el = nodesLayer.querySelector(`[data-node-id="${node.id}"] .node-preview`); if (el) el.textContent = node.lastError ? `오류: ${node.lastError}` : formatValue(node.lastValue, true); }
function invalidatePreviews() { for (const node of graph.nodes.values()) { node.lastValue = undefined; node.lastError = null; updateNodePreview(node); } renderInspector(); }
function formatValue(value, compact = false) {
  if (value === undefined) return '계산 전'; if (typeof value === 'number') return Number.isFinite(value) ? String(Number(value.toFixed(6))) : String(value);
  if (value?.kind === 'image') return '그림';
  if (isArrayValue(value)) { const shape = value.shape.join('×'); if (compact) return `배열 ${shape}`; const sample = Array.from(value.data.slice(0, 18)).map(v => Number(v.toFixed(3))); return `shape: [${value.shape.join(', ')}]\n[${sample.join(', ')}${value.data.length > sample.length ? ', …' : ''}]`; }
  if (value?.kind === 'dataset') return `데이터 ${value.classes.length}종`;
  if (value?.kind === 'datasetClass') return `종류: ${value.name}`;
  if (value?.kind === 'sample') return `${value.className} #${value.index}`;
  return String(value);
}

function updateWires() {
  wiresSvg.setAttribute('viewBox', `0 0 ${workspace.clientWidth} ${workspace.clientHeight}`); wiresSvg.replaceChildren();
  for (const c of graph.connections) { const fromEl = nodesLayer.querySelector(`[data-node-id="${c.from}"] .port.output`), toEl = nodesLayer.querySelector(`[data-node-id="${c.to}"] .port.input[data-port-index="${c.inputIndex}"]`); if (!fromEl || !toEl) continue; const a = portCenter(fromEl), b = portCenter(toEl); const path = document.createElementNS('http://www.w3.org/2000/svg', 'path'); path.setAttribute('class', 'wire'); const bend = Math.max(60, Math.abs(b.x - a.x) * 0.45); path.setAttribute('d', `M ${a.x} ${a.y} C ${a.x + bend} ${a.y}, ${b.x - bend} ${b.y}, ${b.x} ${b.y}`); wiresSvg.appendChild(path); }
}
function portCenter(el) { const wr = workspace.getBoundingClientRect(), r = el.getBoundingClientRect(); return { x: r.left - wr.left + r.width / 2, y: r.top - wr.top + r.height / 2 }; }
function workspacePoint(event) { const rect = workspace.getBoundingClientRect(); return { x: event.clientX - rect.left, y: event.clientY - rect.top }; }
function syncWorkspaceState() { emptyState.hidden = graph.nodes.size > 0; workspaceStatus.textContent = `${graph.nodes.size}개 블록`; }
function resetWorkspace() { graph.nodes.clear(); graph.connections = []; nodesLayer.replaceChildren(); wiresSvg.replaceChildren(); nextNodeId = 1; selectedNodeId = null; cancelGroupSelection(); cancelConnection(); renderInspector(); syncWorkspaceState(); }
function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }
function escapeHtml(text) { return String(text).replace(/[&<>'"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[ch]); }
