// Pan and zoom support for the math-block workspace.
// Node positions stay in world coordinates; only the viewport transform changes.

const WORKSPACE_MIN_ZOOM = 0.35;
const WORKSPACE_MAX_ZOOM = 2.5;
const WORKSPACE_ZOOM_FACTOR = 1.2;
const workspaceView = { x: 0, y: 0, zoom: 1 };

function workspaceLocalPoint(clientX, clientY) {
  const rect = workspace.getBoundingClientRect();
  return { x: clientX - rect.left, y: clientY - rect.top };
}

function applyWorkspaceView() {
  const transform = `translate(${workspaceView.x}px, ${workspaceView.y}px) scale(${workspaceView.zoom})`;
  nodesLayer.style.transformOrigin = '0 0';
  wiresSvg.style.transformOrigin = '0 0';
  nodesLayer.style.transform = transform;
  wiresSvg.style.transform = transform;

  workspace.style.backgroundSize = `${20 * workspaceView.zoom}px ${20 * workspaceView.zoom}px`;
  workspace.style.backgroundPosition = `${workspaceView.x}px ${workspaceView.y}px`;

  const label = document.getElementById('workspaceZoomValue');
  if (label) label.textContent = `${Math.round(workspaceView.zoom * 100)}%`;
}

function setWorkspaceZoom(nextZoom, anchor = null) {
  const zoom = Math.min(WORKSPACE_MAX_ZOOM, Math.max(WORKSPACE_MIN_ZOOM, Number(nextZoom) || 1));
  const rect = workspace.getBoundingClientRect();
  const local = anchor || { x: rect.width / 2, y: rect.height / 2 };
  const worldX = (local.x - workspaceView.x) / workspaceView.zoom;
  const worldY = (local.y - workspaceView.y) / workspaceView.zoom;

  workspaceView.zoom = zoom;
  workspaceView.x = local.x - worldX * zoom;
  workspaceView.y = local.y - worldY * zoom;
  applyWorkspaceView();
  updateWires();
}

function resetWorkspaceView() {
  workspaceView.x = 0;
  workspaceView.y = 0;
  workspaceView.zoom = 1;
  applyWorkspaceView();
  updateWires();
}

function getWorkspaceViewSnapshot() {
  return { x: workspaceView.x, y: workspaceView.y, zoom: workspaceView.zoom };
}

function restoreWorkspaceViewSnapshot(saved) {
  const x = Number(saved?.x);
  const y = Number(saved?.y);
  const zoom = Number(saved?.zoom);
  workspaceView.x = Number.isFinite(x) ? x : 0;
  workspaceView.y = Number.isFinite(y) ? y : 0;
  workspaceView.zoom = Number.isFinite(zoom)
    ? Math.min(WORKSPACE_MAX_ZOOM, Math.max(WORKSPACE_MIN_ZOOM, zoom))
    : 1;
  applyWorkspaceView();
  updateWires();
}

// Convert pointer positions and port centers back into world coordinates.
workspacePoint = function(event) {
  const local = workspaceLocalPoint(event.clientX, event.clientY);
  return {
    x: (local.x - workspaceView.x) / workspaceView.zoom,
    y: (local.y - workspaceView.y) / workspaceView.zoom
  };
};

portCenter = function(el) {
  const wr = workspace.getBoundingClientRect();
  const r = el.getBoundingClientRect();
  const localX = r.left - wr.left + r.width / 2;
  const localY = r.top - wr.top + r.height / 2;
  return {
    x: (localX - workspaceView.x) / workspaceView.zoom,
    y: (localY - workspaceView.y) / workspaceView.zoom
  };
};

function installViewportControls() {
  if (document.getElementById('workspaceViewportControls')) return;
  const controls = document.createElement('div');
  controls.id = 'workspaceViewportControls';
  controls.className = 'workspace-viewport-controls';
  controls.innerHTML = `
    <button type="button" id="workspaceZoomOut" class="ghost" title="축소" aria-label="작업공간 축소">−</button>
    <button type="button" id="workspaceZoomValue" class="ghost workspace-zoom-value" title="100%로 맞추기">100%</button>
    <button type="button" id="workspaceZoomIn" class="ghost" title="확대" aria-label="작업공간 확대">+</button>
    <button type="button" id="workspaceViewReset" class="ghost" title="화면 위치와 배율 초기화">화면 초기화</button>`;
  workspaceStatus.before(controls);

  document.getElementById('workspaceZoomOut').addEventListener('click', () => setWorkspaceZoom(workspaceView.zoom / WORKSPACE_ZOOM_FACTOR));
  document.getElementById('workspaceZoomIn').addEventListener('click', () => setWorkspaceZoom(workspaceView.zoom * WORKSPACE_ZOOM_FACTOR));
  document.getElementById('workspaceZoomValue').addEventListener('click', () => setWorkspaceZoom(1));
  document.getElementById('workspaceViewReset').addEventListener('click', resetWorkspaceView);

  const style = document.createElement('style');
  style.textContent = `
    .workspace { cursor: grab; touch-action: none; }
    .workspace.panning { cursor: grabbing; }
    .nodes, .wires { transform-origin: 0 0; will-change: transform; }
    .workspace-viewport-controls { display: inline-flex; align-items: center; gap: 4px; }
    .workspace-viewport-controls button { min-width: 32px; padding: 6px 8px; }
    .workspace-viewport-controls .workspace-zoom-value { min-width: 58px; font-variant-numeric: tabular-nums; }
  `;
  document.head.appendChild(style);
}

// Capture node drags before the original bounded drag handler so nodes can live
// anywhere in the world coordinate plane, including outside the visible viewport.
workspace.addEventListener('pointerdown', event => {
  const nodeEl = event.target.closest('.node');
  if (nodeEl && nodesLayer.contains(nodeEl)) {
    if (event.target.closest('.port')) return;
    const nodeId = Number(nodeEl.dataset.nodeId);

    if (groupSelectionMode) {
      event.preventDefault();
      event.stopPropagation();
      toggleGroupNode(nodeId);
      return;
    }

    if (event.button != null && event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();

    const node = graph.nodes.get(nodeId);
    if (!node) return;
    selectNode(nodeId);
    cancelConnection();

    const start = workspacePoint(event);
    const origin = { x: node.x, y: node.y };
    nodeEl.classList.add('dragging');
    nodeEl.setPointerCapture?.(event.pointerId);

    const move = e => {
      e.preventDefault();
      const p = workspacePoint(e);
      node.x = origin.x + p.x - start.x;
      node.y = origin.y + p.y - start.y;
      nodeEl.style.left = `${node.x}px`;
      nodeEl.style.top = `${node.y}px`;
      updateWires();
    };

    const finish = e => {
      nodeEl.classList.remove('dragging');
      if (nodeEl.hasPointerCapture?.(e.pointerId)) nodeEl.releasePointerCapture(e.pointerId);
      nodeEl.removeEventListener('pointermove', move);
      nodeEl.removeEventListener('pointerup', finish);
      nodeEl.removeEventListener('pointercancel', finish);
    };

    nodeEl.addEventListener('pointermove', move);
    nodeEl.addEventListener('pointerup', finish);
    nodeEl.addEventListener('pointercancel', finish);
    return;
  }

  if (event.button != null && event.button !== 0) return;
  event.preventDefault();
  cancelConnection();

  const start = { x: event.clientX, y: event.clientY };
  const origin = { x: workspaceView.x, y: workspaceView.y };
  workspace.classList.add('panning');
  workspace.setPointerCapture?.(event.pointerId);

  const move = e => {
    e.preventDefault();
    workspaceView.x = origin.x + e.clientX - start.x;
    workspaceView.y = origin.y + e.clientY - start.y;
    applyWorkspaceView();
  };

  const finish = e => {
    workspace.classList.remove('panning');
    if (workspace.hasPointerCapture?.(e.pointerId)) workspace.releasePointerCapture(e.pointerId);
    workspace.removeEventListener('pointermove', move);
    workspace.removeEventListener('pointerup', finish);
    workspace.removeEventListener('pointercancel', finish);
  };

  workspace.addEventListener('pointermove', move);
  workspace.addEventListener('pointerup', finish);
  workspace.addEventListener('pointercancel', finish);
}, true);

// Ctrl+wheel also covers the pinch gesture emitted by many desktop trackpads.
workspace.addEventListener('wheel', event => {
  if (!event.ctrlKey && !event.metaKey) return;
  event.preventDefault();
  const anchor = workspaceLocalPoint(event.clientX, event.clientY);
  const factor = Math.exp(-event.deltaY * 0.002);
  setWorkspaceZoom(workspaceView.zoom * factor, anchor);
}, { passive: false });

const resetWorkspaceWithoutViewport = resetWorkspace;
resetWorkspace = function() {
  resetWorkspaceWithoutViewport();
  resetWorkspaceView();
};

installViewportControls();
applyWorkspaceView();
