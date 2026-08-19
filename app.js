const workspace = document.getElementById('workspace');
const nodesLayer = document.getElementById('nodes');
const wiresSvg = document.getElementById('wires');
const emptyState = document.getElementById('emptyState');
const workspaceStatus = document.getElementById('workspaceStatus');
const connectionHint = document.getElementById('connectionHint');
const evaluateBtn = document.getElementById('evaluateBtn');
const resetWorkspaceBtn = document.getElementById('resetWorkspaceBtn');

const inspectorEmpty = document.getElementById('inspectorEmpty');
const inspectorContent = document.getElementById('inspectorContent');
const inspectorTitle = document.getElementById('inspectorTitle');
const inspectorDescription = document.getElementById('inspectorDescription');
const inspectorFormula = document.getElementById('inspectorFormula');
const inspectorValue = document.getElementById('inspectorValue');
const inspectorControls = document.getElementById('inspectorControls');
const deleteNodeBtn = document.getElementById('deleteNodeBtn');

const drawCanvas = document.getElementById('drawCanvas');
const drawCtx = drawCanvas.getContext('2d', { willReadFrequently: true });
const previewCanvas = document.getElementById('previewCanvas');
const previewCtx = previewCanvas.getContext('2d');
const clearDrawBtn = document.getElementById('clearDrawBtn');

let nextNodeId = 1;
let selectedNodeId = null;
let pendingOutput = null;
let drawing = false;
let lastPoint = { x: 0, y: 0 };

const graph = {
  nodes: new Map(),
  connections: []
};

const BLOCKS = {
  number: {
    title: '숫자', kind: 'source', inputs: [],
    description: '하나의 실수 값을 만든다.',
    formula: node => String(node.params.value),
    controls: [{ key: 'value', label: '값', type: 'number', step: 'any', default: 1 }],
    compute: node => Number(node.params.value)
  },
  drawing: {
    title: '그림 입력', kind: 'source', inputs: [],
    description: '오른쪽 그림판의 현재 그림을 입력으로 사용한다.',
    formula: () => 'image',
    compute: () => ({ kind: 'image', canvas: drawCanvas })
  },
  resize28: {
    title: '28×28 변환', kind: 'transform', inputs: ['image'],
    description: '그림의 잉크 영역을 찾아 비율을 유지한 채 28×28 행렬로 바꾼다.',
    formula: () => 'T_{28×28}(image)',
    compute: (node, [input]) => {
      if (!input || input.kind !== 'image') throw new Error('그림 입력이 필요합니다.');
      const value = preprocessCanvasTo28(input.canvas);
      renderPreview(value.data);
      return value;
    }
  },
  flatten: {
    title: '펼치기', kind: 'transform', inputs: ['x'],
    description: '여러 차원의 배열을 한 줄의 벡터로 펼친다.',
    formula: () => 'flatten(x)',
    compute: (node, [input]) => {
      const arr = asArrayValue(input);
      return { kind: 'array', data: new Float32Array(arr.data), shape: [arr.data.length] };
    }
  },
  add: {
    title: '더하기', kind: 'operation', inputs: ['a', 'b'],
    description: '두 수 또는 배열을 더한다. 수와 배열은 자동으로 브로드캐스트한다.',
    formula: () => 'a + b',
    compute: (node, [a, b]) => elementwiseBinary(a, b, (x, y) => x + y)
  },
  multiply: {
    title: '곱하기', kind: 'operation', inputs: ['a', 'b'],
    description: '두 수 또는 배열을 곱한다. 현재는 원소별 곱셈이다.',
    formula: () => 'a × b',
    compute: (node, [a, b]) => elementwiseBinary(a, b, (x, y) => x * y)
  },
  square: {
    title: '제곱', kind: 'operation', inputs: ['x'],
    description: '입력의 각 값을 제곱한다.',
    formula: () => 'x²',
    compute: (node, [x]) => elementwiseUnary(x, v => v * v)
  },
  sum: {
    title: '합', kind: 'operation', inputs: ['x'],
    description: '배열의 모든 원소를 더해 하나의 수로 만든다.',
    formula: () => 'Σxᵢ',
    compute: (node, [x]) => {
      if (typeof x === 'number') return x;
      const arr = asArrayValue(x);
      let sum = 0;
      for (const value of arr.data) sum += value;
      return sum;
    }
  },
  mean: {
    title: '평균', kind: 'operation', inputs: ['x'],
    description: '배열 원소의 산술평균을 계산한다.',
    formula: () => '(1/n)Σxᵢ',
    compute: (node, [x]) => {
      if (typeof x === 'number') return x;
      const arr = asArrayValue(x);
      if (!arr.data.length) return 0;
      let sum = 0;
      for (const value of arr.data) sum += value;
      return sum / arr.data.length;
    }
  },
  relu: {
    title: 'ReLU', kind: 'function', inputs: ['x'],
    description: '0보다 작으면 0, 크면 입력값을 그대로 돌려주는 함수.',
    formula: () => 'max(0, x)',
    compute: (node, [x]) => elementwiseUnary(x, v => Math.max(0, v))
  },
  display: {
    title: '값 보기', kind: 'sink', inputs: ['x'],
    description: '연결된 블록의 계산 결과를 그대로 보여준다.',
    formula: () => 'x',
    compute: (node, [x]) => x
  }
};

function asArrayValue(value) {
  if (!value || value.kind !== 'array') throw new Error('배열 입력이 필요합니다.');
  return value;
}

function elementwiseUnary(value, fn) {
  if (typeof value === 'number') return fn(value);
  const arr = asArrayValue(value);
  const data = new Float32Array(arr.data.length);
  for (let i = 0; i < data.length; i++) data[i] = fn(arr.data[i]);
  return { kind: 'array', data, shape: [...arr.shape] };
}

function elementwiseBinary(a, b, fn) {
  if (typeof a === 'number' && typeof b === 'number') return fn(a, b);
  if (typeof a === 'number' && b?.kind === 'array') return elementwiseUnary(b, x => fn(a, x));
  if (a?.kind === 'array' && typeof b === 'number') return elementwiseUnary(a, x => fn(x, b));
  const aa = asArrayValue(a);
  const bb = asArrayValue(b);
  if (aa.data.length !== bb.data.length) throw new Error('배열의 원소 수가 서로 다릅니다.');
  const data = new Float32Array(aa.data.length);
  for (let i = 0; i < data.length; i++) data[i] = fn(aa.data[i], bb.data[i]);
  return { kind: 'array', data, shape: [...aa.shape] };
}

function addBlock(type) {
  const def = BLOCKS[type];
  const id = nextNodeId++;
  const count = graph.nodes.size;
  const node = { id, type, x: 34 + (count % 4) * 205, y: 34 + Math.floor(count / 4) * 135, params: {} };
  for (const control of def.controls || []) node.params[control.key] = control.default;
  graph.nodes.set(id, node);
  renderNode(node);
  selectNode(id);
  syncWorkspaceState();
  updateWires();
}

function renderNode(node) {
  const def = BLOCKS[node.type];
  const el = document.createElement('div');
  el.className = `node ${def.kind}`;
  el.dataset.nodeId = node.id;
  el.style.left = `${node.x}px`;
  el.style.top = `${node.y}px`;

  const inputsHtml = def.inputs.map((label, index) => `
    <div class="port-row">
      <span class="port input" data-node-id="${node.id}" data-port-index="${index}" title="입력 ${label}"></span>
      <span class="port-label">${label}</span>
    </div>`).join('');

  el.innerHTML = `
    <div class="node-head">
      <span>${def.title}</span>
      <span class="node-kind">${kindLabel(def.kind)}</span>
    </div>
    <div class="node-body">
      ${inputsHtml}
      ${def.inputs.length ? '' : '<span class="muted">입력 없음</span>'}
      <div class="node-preview">계산 전</div>
      <div class="port-row output-row">
        <span class="port-label">출력</span>
        <span class="port output" data-node-id="${node.id}" title="출력"></span>
      </div>
    </div>`;

  nodesLayer.appendChild(el);
  installNodeEvents(el, node);
}

function kindLabel(kind) {
  return ({ source: '입력', transform: '변환', operation: '연산', function: '함수', sink: '확인' })[kind] || kind;
}

function installNodeEvents(el, node) {
  el.addEventListener('pointerdown', event => {
    if (event.target.classList.contains('port')) return;
    selectNode(node.id);
  });

  const head = el.querySelector('.node-head');
  head.addEventListener('pointerdown', event => {
    event.preventDefault();
    selectNode(node.id);
    const start = workspacePoint(event);
    const origin = { x: node.x, y: node.y };
    head.setPointerCapture(event.pointerId);

    const move = e => {
      const p = workspacePoint(e);
      node.x = clamp(origin.x + p.x - start.x, 0, workspace.clientWidth - el.offsetWidth);
      node.y = clamp(origin.y + p.y - start.y, 0, workspace.clientHeight - el.offsetHeight);
      el.style.left = `${node.x}px`;
      el.style.top = `${node.y}px`;
      updateWires();
    };
    const up = e => {
      head.releasePointerCapture?.(e.pointerId);
      head.removeEventListener('pointermove', move);
      head.removeEventListener('pointerup', up);
      head.removeEventListener('pointercancel', up);
    };
    head.addEventListener('pointermove', move);
    head.addEventListener('pointerup', up);
    head.addEventListener('pointercancel', up);
  });

  el.querySelector('.port.output').addEventListener('click', event => {
    event.stopPropagation();
    beginConnection(node.id, event.currentTarget);
  });

  el.querySelectorAll('.port.input').forEach(port => {
    port.addEventListener('click', event => {
      event.stopPropagation();
      finishConnection(node.id, Number(port.dataset.portIndex));
    });
  });
}

function beginConnection(nodeId, portEl) {
  clearHotPorts();
  pendingOutput = nodeId;
  portEl.classList.add('hot');
  document.querySelectorAll('.port.input').forEach(port => port.classList.add('hot'));
  connectionHint.textContent = '연결할 블록의 왼쪽 입력 ○를 누르세요.';
}

function finishConnection(toNodeId, inputIndex) {
  if (pendingOutput == null) return;
  if (pendingOutput === toNodeId) return cancelConnection();
  graph.connections = graph.connections.filter(c => !(c.to === toNodeId && c.inputIndex === inputIndex));
  graph.connections.push({ from: pendingOutput, to: toNodeId, inputIndex });
  cancelConnection();
  updateWires();
  invalidatePreviews();
}

function cancelConnection() {
  pendingOutput = null;
  clearHotPorts();
  connectionHint.textContent = '블록을 놓고 수학식을 직접 만드세요.';
}

function clearHotPorts() {
  document.querySelectorAll('.port.hot').forEach(el => el.classList.remove('hot'));
}

function deleteSelectedNode() {
  if (selectedNodeId == null) return;
  graph.nodes.delete(selectedNodeId);
  graph.connections = graph.connections.filter(c => c.from !== selectedNodeId && c.to !== selectedNodeId);
  nodesLayer.querySelector(`[data-node-id="${selectedNodeId}"]`)?.remove();
  selectedNodeId = null;
  renderInspector();
  syncWorkspaceState();
  updateWires();
}

function selectNode(id) {
  selectedNodeId = id;
  nodesLayer.querySelectorAll('.node').forEach(el => {
    el.classList.toggle('selected', Number(el.dataset.nodeId) === id);
  });
  renderInspector();
}

function renderInspector() {
  const node = selectedNodeId == null ? null : graph.nodes.get(selectedNodeId);
  inspectorEmpty.hidden = Boolean(node);
  inspectorContent.hidden = !node;
  deleteNodeBtn.disabled = !node;
  if (!node) return;

  const def = BLOCKS[node.type];
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
    const input = document.createElement('input');
    input.type = control.type;
    input.step = control.step || '1';
    input.value = node.params[control.key];
    input.addEventListener('input', () => {
      node.params[control.key] = input.value;
      inspectorFormula.textContent = def.formula(node);
      invalidatePreviews();
    });
    row.append(label, input);
    inspectorControls.appendChild(row);
  }
}

function evaluateGraph() {
  const memo = new Map();
  const visiting = new Set();
  let successCount = 0;
  for (const node of graph.nodes.values()) { node.lastValue = undefined; node.lastError = null; }

  for (const node of graph.nodes.values()) {
    try {
      node.lastValue = evaluateNode(node.id, memo, visiting);
      successCount++;
    } catch (error) {
      node.lastError = error.message;
    }
    updateNodePreview(node);
  }
  renderInspector();
  workspaceStatus.textContent = `${successCount}/${graph.nodes.size} 계산`;
}

function evaluateNode(nodeId, memo, visiting) {
  if (memo.has(nodeId)) return memo.get(nodeId);
  if (visiting.has(nodeId)) throw new Error('순환 연결은 계산할 수 없습니다.');
  const node = graph.nodes.get(nodeId);
  if (!node) throw new Error('존재하지 않는 블록입니다.');
  const def = BLOCKS[node.type];
  visiting.add(nodeId);

  const inputs = def.inputs.map((label, inputIndex) => {
    const connection = graph.connections.find(c => c.to === nodeId && c.inputIndex === inputIndex);
    if (!connection) throw new Error(`입력 '${label}'이 연결되지 않았습니다.`);
    return evaluateNode(connection.from, memo, visiting);
  });

  const value = def.compute(node, inputs);
  memo.set(nodeId, value);
  visiting.delete(nodeId);
  return value;
}

function updateNodePreview(node) {
  const el = nodesLayer.querySelector(`[data-node-id="${node.id}"] .node-preview`);
  if (!el) return;
  el.textContent = node.lastError ? `오류: ${node.lastError}` : formatValue(node.lastValue, true);
}

function invalidatePreviews() {
  for (const node of graph.nodes.values()) {
    node.lastValue = undefined;
    node.lastError = null;
    updateNodePreview(node);
  }
  renderInspector();
}

function formatValue(value, compact = false) {
  if (value === undefined) return '계산 전';
  if (typeof value === 'number') return Number.isFinite(value) ? String(Number(value.toFixed(6))) : String(value);
  if (value?.kind === 'image') return '그림';
  if (value?.kind === 'array') {
    const shape = value.shape.join('×');
    if (compact) return `배열 ${shape}`;
    const sample = Array.from(value.data.slice(0, 18)).map(v => Number(v.toFixed(3)));
    return `shape: [${value.shape.join(', ')}]\n[${sample.join(', ')}${value.data.length > sample.length ? ', …' : ''}]`;
  }
  return String(value);
}

function updateWires() {
  wiresSvg.setAttribute('viewBox', `0 0 ${workspace.clientWidth} ${workspace.clientHeight}`);
  wiresSvg.replaceChildren();
  for (const c of graph.connections) {
    const fromEl = nodesLayer.querySelector(`[data-node-id="${c.from}"] .port.output`);
    const toEl = nodesLayer.querySelector(`[data-node-id="${c.to}"] .port.input[data-port-index="${c.inputIndex}"]`);
    if (!fromEl || !toEl) continue;
    const a = portCenter(fromEl);
    const b = portCenter(toEl);
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('class', 'wire');
    const bend = Math.max(60, Math.abs(b.x - a.x) * 0.45);
    path.setAttribute('d', `M ${a.x} ${a.y} C ${a.x + bend} ${a.y}, ${b.x - bend} ${b.y}, ${b.x} ${b.y}`);
    wiresSvg.appendChild(path);
  }
}

function portCenter(el) {
  const wr = workspace.getBoundingClientRect();
  const r = el.getBoundingClientRect();
  return { x: r.left - wr.left + r.width / 2, y: r.top - wr.top + r.height / 2 };
}

function workspacePoint(event) {
  const rect = workspace.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function syncWorkspaceState() {
  emptyState.hidden = graph.nodes.size > 0;
  workspaceStatus.textContent = `${graph.nodes.size}개 블록`;
}

function resetWorkspace() {
  graph.nodes.clear();
  graph.connections = [];
  nodesLayer.replaceChildren();
  wiresSvg.replaceChildren();
  nextNodeId = 1;
  selectedNodeId = null;
  cancelConnection();
  renderInspector();
  syncWorkspaceState();
}

function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }

function resetDrawCanvas() {
  drawCtx.fillStyle = '#fff';
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  drawCtx.lineCap = 'round';
  drawCtx.lineJoin = 'round';
  drawCtx.lineWidth = 18;
  drawCtx.strokeStyle = '#111';
  resetPreview();
}

function resetPreview() {
  previewCtx.fillStyle = '#000';
  previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
}

function pointerOnDraw(event) {
  const rect = drawCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * drawCanvas.width / rect.width,
    y: (event.clientY - rect.top) * drawCanvas.height / rect.height
  };
}

function findInkBounds(ctx, canvas) {
  const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
  let minX = canvas.width, minY = canvas.height, maxX = -1, maxY = -1;
  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      const i = (y * canvas.width + x) * 4;
      const gray = (image.data[i] + image.data[i + 1] + image.data[i + 2]) / 3;
      if (gray < 245) {
        minX = Math.min(minX, x); minY = Math.min(minY, y);
        maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
      }
    }
  }
  return maxX < minX ? null : { minX, minY, maxX, maxY };
}

function preprocessCanvasTo28(canvas) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const bounds = findInkBounds(ctx, canvas);
  const temp = document.createElement('canvas');
  temp.width = 28;
  temp.height = 28;
  const tctx = temp.getContext('2d', { willReadFrequently: true });
  tctx.fillStyle = '#fff';
  tctx.fillRect(0, 0, 28, 28);

  if (bounds) {
    const cropW = bounds.maxX - bounds.minX + 1;
    const cropH = bounds.maxY - bounds.minY + 1;
    const inner = 20;
    const scale = Math.min(inner / cropW, inner / cropH);
    const w = Math.max(1, Math.round(cropW * scale));
    const h = Math.max(1, Math.round(cropH * scale));
    const dx = Math.floor((28 - w) / 2);
    const dy = Math.floor((28 - h) / 2);
    tctx.imageSmoothingEnabled = true;
    tctx.drawImage(canvas, bounds.minX, bounds.minY, cropW, cropH, dx, dy, w, h);
  }

  const small = tctx.getImageData(0, 0, 28, 28);
  const data = new Float32Array(784);
  for (let i = 0; i < data.length; i++) {
    const p = i * 4;
    const gray = (small.data[p] + small.data[p + 1] + small.data[p + 2]) / 3;
    data[i] = 1 - gray / 255;
  }
  return { kind: 'array', data, shape: [28, 28] };
}

function renderPreview(data) {
  const tiny = document.createElement('canvas');
  tiny.width = 28;
  tiny.height = 28;
  const tctx = tiny.getContext('2d');
  const image = tctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = Math.round(clamp(data[i], 0, 1) * 255);
    const p = i * 4;
    image.data[p] = v;
    image.data[p + 1] = v;
    image.data[p + 2] = v;
    image.data[p + 3] = 255;
  }
  tctx.putImageData(image, 0, 0);
  previewCtx.imageSmoothingEnabled = false;
  previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  previewCtx.drawImage(tiny, 0, 0, previewCanvas.width, previewCanvas.height);
}

for (const button of document.querySelectorAll('[data-block]')) {
  button.addEventListener('click', () => addBlock(button.dataset.block));
}

evaluateBtn.addEventListener('click', evaluateGraph);
resetWorkspaceBtn.addEventListener('click', resetWorkspace);
deleteNodeBtn.addEventListener('click', deleteSelectedNode);
clearDrawBtn.addEventListener('click', () => { resetDrawCanvas(); invalidatePreviews(); });
workspace.addEventListener('click', event => {
  if (event.target === workspace || event.target === nodesLayer || event.target === emptyState) cancelConnection();
});
window.addEventListener('resize', updateWires);

drawCanvas.addEventListener('pointerdown', event => {
  drawing = true;
  lastPoint = pointerOnDraw(event);
  drawCanvas.setPointerCapture?.(event.pointerId);
});
drawCanvas.addEventListener('pointermove', event => {
  if (!drawing) return;
  const p = pointerOnDraw(event);
  drawCtx.beginPath();
  drawCtx.moveTo(lastPoint.x, lastPoint.y);
  drawCtx.lineTo(p.x, p.y);
  drawCtx.stroke();
  lastPoint = p;
});
function stopDraw(event) {
  drawing = false;
  if (event?.pointerId != null && drawCanvas.hasPointerCapture?.(event.pointerId)) drawCanvas.releasePointerCapture(event.pointerId);
  invalidatePreviews();
}
drawCanvas.addEventListener('pointerup', stopDraw);
drawCanvas.addEventListener('pointercancel', stopDraw);

resetDrawCanvas();
resetWorkspace();
