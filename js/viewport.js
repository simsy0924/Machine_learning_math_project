// Pan, zoom and node dragging for the workspace.
// Node positions stay in world coordinates; only the viewport transform changes,
// so workspacePoint/portCenter convert pointer and DOM positions back to world
// space for everything else in the app.

const WORKSPACE_MIN_ZOOM = 0.35;
const WORKSPACE_MAX_ZOOM = 2.5;
const WORKSPACE_ZOOM_FACTOR = 1.2;
const workspaceView = { x: 0, y: 0, zoom: 1 };

function workspaceLocalPoint(clientX, clientY) {
  const rect = workspace.getBoundingClientRect();
  return { x: clientX - rect.left, y: clientY - rect.top };
}

function clampWorkspaceZoom(value) {
  return Math.min(WORKSPACE_MAX_ZOOM, Math.max(WORKSPACE_MIN_ZOOM, Number(value) || 1));
}

function workspacePoint(event) {
  const local = workspaceLocalPoint(event.clientX, event.clientY);
  return {
    x: (local.x - workspaceView.x) / workspaceView.zoom,
    y: (local.y - workspaceView.y) / workspaceView.zoom
  };
}

function portCenter(el) {
  const wr = workspace.getBoundingClientRect();
  const r = el.getBoundingClientRect();
  const localX = r.left - wr.left + r.width / 2;
  const localY = r.top - wr.top + r.height / 2;
  return {
    x: (localX - workspaceView.x) / workspaceView.zoom,
    y: (localY - workspaceView.y) / workspaceView.zoom
  };
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
  const zoom = clampWorkspaceZoom(nextZoom);
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
  workspaceView.zoom = Number.isFinite(zoom) ? clampWorkspaceZoom(zoom) : 1;
  applyWorkspaceView();
  updateWires();
}

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

// One-pointer panning/node dragging and two-pointer touch pinch zoom share the
// same pointer state so switching between them is smooth on tablets and phones.
const workspaceTouches = new Map();
let workspaceGesture = null;
let activeNodeDrag = null;

function touchPair() {
  return [...workspaceTouches.values()].slice(0, 2);
}

function pairGeometry(points) {
  const [a, b] = points;
  const dx = b.clientX - a.clientX;
  const dy = b.clientY - a.clientY;
  return {
    distance: Math.max(1, Math.hypot(dx, dy)),
    midpoint: workspaceLocalPoint((a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2)
  };
}

function stopNodeDrag() {
  if (!activeNodeDrag) return;
  activeNodeDrag.nodeEl.classList.remove('dragging');
  activeNodeDrag = null;
}

// Called by resetWorkspace: drop any gesture in flight along with the graph.
function resetWorkspaceViewState() {
  workspaceTouches.clear();
  workspaceGesture = null;
  stopNodeDrag();
  resetWorkspaceView();
}

function startPinchGesture() {
  const points = touchPair();
  if (points.length < 2) return;
  stopNodeDrag();
  const geometry = pairGeometry(points);
  const anchorWorld = {
    x: (geometry.midpoint.x - workspaceView.x) / workspaceView.zoom,
    y: (geometry.midpoint.y - workspaceView.y) / workspaceView.zoom
  };
  workspaceGesture = {
    mode: 'pinch',
    startDistance: geometry.distance,
    startZoom: workspaceView.zoom,
    anchorWorld
  };
  workspace.classList.add('panning');
}

function startPanFromPointer(pointerId, point) {
  workspaceGesture = {
    mode: 'pan',
    pointerId,
    startClientX: point.clientX,
    startClientY: point.clientY,
    originX: workspaceView.x,
    originY: workspaceView.y
  };
  workspace.classList.add('panning');
}

workspace.addEventListener('pointerdown', event => {
  const isTouch = event.pointerType === 'touch';
  if (event.button != null && event.button !== 0 && !isTouch) return;

  if (isTouch) {
    workspaceTouches.set(event.pointerId, { pointerId: event.pointerId, clientX: event.clientX, clientY: event.clientY });
    workspace.setPointerCapture?.(event.pointerId);

    if (workspaceTouches.size >= 2) {
      event.preventDefault();
      event.stopPropagation();
      cancelConnection();
      startPinchGesture();
      return;
    }
  }

  const nodeEl = event.target.closest('.node');
  if (nodeEl && nodesLayer.contains(nodeEl)) {
    if (event.target.closest('.port')) {
      if (isTouch) workspaceTouches.delete(event.pointerId);
      return;
    }

    const nodeId = Number(nodeEl.dataset.nodeId);
    if (groupSelectionMode) {
      event.preventDefault();
      event.stopPropagation();
      toggleGroupNode(nodeId);
      return;
    }

    event.preventDefault();
    event.stopPropagation();
    const node = graph.nodes.get(nodeId);
    if (!node) return;

    selectNode(nodeId);
    cancelConnection();
    const start = workspacePoint(event);
    activeNodeDrag = {
      pointerId: event.pointerId,
      node,
      nodeEl,
      startX: start.x,
      startY: start.y,
      originX: node.x,
      originY: node.y
    };
    nodeEl.classList.add('dragging');
    workspace.setPointerCapture?.(event.pointerId);
    return;
  }

  event.preventDefault();
  event.stopPropagation();
  cancelConnection();
  startPanFromPointer(event.pointerId, { clientX: event.clientX, clientY: event.clientY });
  workspace.setPointerCapture?.(event.pointerId);
}, true);

workspace.addEventListener('pointermove', event => {
  if (event.pointerType === 'touch' && workspaceTouches.has(event.pointerId)) {
    workspaceTouches.set(event.pointerId, { pointerId: event.pointerId, clientX: event.clientX, clientY: event.clientY });
  }

  if (workspaceGesture?.mode === 'pinch' && workspaceTouches.size >= 2) {
    event.preventDefault();
    const geometry = pairGeometry(touchPair());
    const zoom = clampWorkspaceZoom(workspaceGesture.startZoom * geometry.distance / workspaceGesture.startDistance);
    workspaceView.zoom = zoom;
    workspaceView.x = geometry.midpoint.x - workspaceGesture.anchorWorld.x * zoom;
    workspaceView.y = geometry.midpoint.y - workspaceGesture.anchorWorld.y * zoom;
    applyWorkspaceView();
    updateWires();
    return;
  }

  if (activeNodeDrag?.pointerId === event.pointerId) {
    event.preventDefault();
    const p = workspacePoint(event);
    const { node, nodeEl } = activeNodeDrag;
    node.x = activeNodeDrag.originX + p.x - activeNodeDrag.startX;
    node.y = activeNodeDrag.originY + p.y - activeNodeDrag.startY;
    nodeEl.style.left = `${node.x}px`;
    nodeEl.style.top = `${node.y}px`;
    updateWires();
    return;
  }

  if (workspaceGesture?.mode === 'pan' && workspaceGesture.pointerId === event.pointerId) {
    event.preventDefault();
    workspaceView.x = workspaceGesture.originX + event.clientX - workspaceGesture.startClientX;
    workspaceView.y = workspaceGesture.originY + event.clientY - workspaceGesture.startClientY;
    applyWorkspaceView();
  }
}, true);

function finishWorkspacePointer(event) {
  const wasTouch = event.pointerType === 'touch';
  if (wasTouch) workspaceTouches.delete(event.pointerId);

  if (activeNodeDrag?.pointerId === event.pointerId) stopNodeDrag();

  if (workspaceGesture?.mode === 'pinch') {
    if (workspaceTouches.size >= 2) {
      startPinchGesture();
    } else if (workspaceTouches.size === 1) {
      const remaining = [...workspaceTouches.values()][0];
      startPanFromPointer(remaining.pointerId, remaining);
    } else {
      workspaceGesture = null;
      workspace.classList.remove('panning');
    }
  } else if (workspaceGesture?.mode === 'pan' && workspaceGesture.pointerId === event.pointerId) {
    workspaceGesture = null;
    workspace.classList.remove('panning');
  }

  if (workspace.hasPointerCapture?.(event.pointerId)) workspace.releasePointerCapture(event.pointerId);
}

workspace.addEventListener('pointerup', finishWorkspacePointer, true);
workspace.addEventListener('pointercancel', finishWorkspacePointer, true);

// Ctrl+wheel also covers the pinch gesture emitted by many desktop trackpads.
workspace.addEventListener('wheel', event => {
  if (!event.ctrlKey && !event.metaKey) return;
  event.preventDefault();
  const anchor = workspaceLocalPoint(event.clientX, event.clientY);
  const factor = Math.exp(-event.deltaY * 0.002);
  setWorkspaceZoom(workspaceView.zoom * factor, anchor);
}, { passive: false });

installViewportControls();
applyWorkspaceView();
