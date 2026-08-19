const BLOCKS = {
  number: {
    title: '숫자', kind: 'source', inputs: [], description: '하나의 실수 값을 만든다.',
    formula: n => String(n.params.value),
    controls: [{ key: 'value', label: '값', type: 'number', step: 'any', default: 1 }],
    compute: n => Number(n.params.value)
  },
  variable: {
    title: '변수', kind: 'source', inputs: [], description: '이름과 값을 가진 변수. 미분할 대상을 지정할 때 사용한다.',
    formula: n => n.params.mode === 'vector' ? `${n.params.name} ∈ R^${n.params.length}` : `${n.params.name} = ${n.params.value}`,
    controls: [
      { key: 'name', label: '변수 이름', type: 'text', default: 'x' },
      { key: 'mode', label: '형태', type: 'select', default: 'scalar', options: [{ value: 'scalar', label: '숫자' }, { value: 'vector', label: '벡터' }] },
      { key: 'value', label: '숫자 값 / 벡터 채움값', type: 'number', step: 'any', default: 1 },
      { key: 'length', label: '벡터 길이', type: 'number', step: '1', default: 3 },
      { key: 'init', label: '벡터 초기값', type: 'select', default: 'constant', options: [{ value: 'constant', label: '같은 값' }, { value: 'random', label: '무작위(-1~1)' }] },
      { key: 'seed', label: '무작위 시드', type: 'number', step: '1', default: 1 }
    ],
    compute: n => {
      if (n.params.mode !== 'vector') return Number(n.params.value);
      const length = Math.max(1, Math.floor(Number(n.params.length) || 1));
      if (n.params.init === 'random') return deterministicRandomVector(length, n.params.seed);
      const data = new Float32Array(length); data.fill(Number(n.params.value)); return arrayValue(data, [length]);
    }
  },
  drawing: {
    title: '그림 입력', kind: 'source', inputs: [], description: '오른쪽 그림판의 현재 그림을 입력으로 사용한다.',
    formula: () => 'image', compute: () => ({ kind: 'image', canvas: drawCanvas })
  },
  dataset: {
    title: '불러온 데이터', kind: 'data', inputs: [], description: '오른쪽에서 현재 불러온 Quick Draw 종류들을 가져온다.',
    formula: () => 'D',
    compute: () => {
      const api = window.quickDrawDataset;
      if (!api) throw new Error('데이터 로더가 아직 준비되지 않았습니다.');
      const classes = api.getLoadedClassNames();
      if (!classes.length) throw new Error('오른쪽에서 Quick Draw 데이터를 먼저 불러오세요.');
      return { kind: 'dataset', classes };
    }
  },
  chooseClass: {
    title: '종류 고르기', kind: 'data', inputs: ['데이터'], description: '불러온 데이터 중 한 종류를 고른다.',
    formula: n => `class = ${n.params.className}`,
    controls: [{ key: 'className', label: '종류', type: 'select', default: 'cat', options: classOptions() }],
    compute: (n, [dataset]) => {
      if (!dataset || dataset.kind !== 'dataset') throw new Error('불러온 데이터 블록이 필요합니다.');
      const name = n.params.className;
      const classIndex = dataset.classes.indexOf(name);
      if (classIndex < 0) throw new Error(`'${name}' 종류가 현재 로드되어 있지 않습니다.`);
      return { kind: 'datasetClass', name, classIndex, classes: [...dataset.classes] };
    }
  },
  chooseSample: {
    title: '샘플 고르기', kind: 'data', inputs: ['종류'], description: '선택한 종류에서 한 장의 샘플을 고른다.',
    formula: n => `sample[${n.params.index}]`,
    controls: [{ key: 'index', label: '샘플 번호 (0~9999)', type: 'number', step: '1', default: 0 }],
    compute: (n, [klass]) => {
      if (!klass || klass.kind !== 'datasetClass') throw new Error('종류 고르기 블록이 필요합니다.');
      const api = window.quickDrawDataset;
      const index = Math.floor(Number(n.params.index));
      const image = api.getNormalizedImage(klass.name, index);
      return { kind: 'sample', className: klass.name, classIndex: klass.classIndex, classes: [...klass.classes], index, image };
    }
  },
  sampleImage: {
    title: '그림값', kind: 'data', inputs: ['샘플'], description: '샘플의 28×28 픽셀을 0~1 숫자 배열로 꺼낸다.',
    formula: () => 'x ∈ R^(28×28)',
    compute: (n, [sample]) => {
      if (!sample || sample.kind !== 'sample') throw new Error('샘플 고르기 블록이 필요합니다.');
      return arrayValue(new Float32Array(sample.image), [28, 28]);
    }
  },
  sampleLabel: {
    title: '종류 번호', kind: 'data', inputs: ['샘플'], description: '현재 로드된 종류 순서에서 이 샘플의 번호를 꺼낸다.',
    formula: () => 'y',
    compute: (n, [sample]) => {
      if (!sample || sample.kind !== 'sample') throw new Error('샘플 고르기 블록이 필요합니다.');
      return sample.classIndex;
    }
  },
  sampleOneHot: {
    title: '종류 벡터', kind: 'data', inputs: ['샘플'], description: '정답 종류만 1이고 나머지는 0인 벡터를 만든다.',
    formula: () => '[0,…,1,…,0]',
    compute: (n, [sample]) => {
      if (!sample || sample.kind !== 'sample') throw new Error('샘플 고르기 블록이 필요합니다.');
      const data = new Float32Array(sample.classes.length); data[sample.classIndex] = 1; return arrayValue(data, [data.length]);
    }
  },
  constantVector: {
    title: '같은 값 벡터', kind: 'generator', inputs: [], description: '같은 숫자로 채운 벡터를 만든다.',
    formula: n => `[${n.params.value}, …] (${n.params.length})`,
    controls: [{ key: 'length', label: '길이', type: 'number', step: '1', default: 3 }, { key: 'value', label: '값', type: 'number', step: 'any', default: 0 }],
    compute: n => { const len = Math.max(1, Math.floor(Number(n.params.length) || 1)); const d = new Float32Array(len); d.fill(Number(n.params.value)); return arrayValue(d, [len]); }
  },
  randomVector: {
    title: '무작위 벡터', kind: 'generator', inputs: [], description: '-1~1 사이의 고정된 무작위 벡터를 만든다.',
    formula: n => `random(${n.params.length}, seed=${n.params.seed})`,
    controls: [{ key: 'length', label: '길이', type: 'number', step: '1', default: 3 }, { key: 'seed', label: '시드', type: 'number', step: '1', default: 1 }],
    compute: n => deterministicRandomVector(Math.max(1, Math.floor(Number(n.params.length) || 1)), n.params.seed)
  },
  resize28: {
    title: '28×28 변환', kind: 'transform', inputs: ['그림'], description: '그림판 입력을 28×28 값으로 바꾼다.',
    formula: () => 'T_(28×28)(image)',
    compute: (n, [input]) => { if (!input || input.kind !== 'image') throw new Error('그림 입력이 필요합니다.'); const value = preprocessCanvasTo28(input.canvas); renderPreview(value.data); return value; }
  },
  flatten: {
    title: '펼치기', kind: 'transform', inputs: ['x'], description: '배열을 한 줄의 벡터로 펼친다.', formula: () => 'flatten(x)',
    compute: (n, [input]) => { const arr = asArrayValue(input); return arrayValue(new Float32Array(arr.data), [arr.data.length]); }
  },
  add: op2('더하기', 'a + b', (a, b) => addValues(a, b)),
  subtract: op2('빼기', 'a - b', (a, b) => subtractValues(a, b)),
  multiply: op2('곱하기', 'a × b', (a, b) => multiplyValues(a, b)),
  divide: op2('나누기', 'a ÷ b', (a, b) => divideValues(a, b)),
  square: op1('제곱', 'x²', x => elementwiseUnary(x, v => v * v)),
  abs: op1('절댓값', '|x|', x => elementwiseUnary(x, v => Math.abs(v))),
  maximum: op2('최댓값', 'max(a,b)', (a, b) => elementwiseBinary(a, b, (x, y) => Math.max(x, y))),
  exp: op1('지수', 'eˣ', x => elementwiseUnary(x, v => Math.exp(v))),
  log: op1('로그', 'ln(x)', x => elementwiseUnary(x, v => Math.log(v))),
  sum: {
    title: '합', kind: 'operation', inputs: ['x'], description: '배열의 원소를 모두 더한다.', formula: () => 'Σxᵢ',
    compute: (n, [x]) => typeof x === 'number' ? x : sumArray(asArrayValue(x))
  },
  mean: {
    title: '평균', kind: 'operation', inputs: ['x'], description: '배열의 산술평균을 구한다.', formula: () => '(1/n)Σxᵢ',
    compute: (n, [x]) => { if (typeof x === 'number') return x; const a = asArrayValue(x); return a.data.length ? sumArray(a) / a.data.length : 0; }
  },
  dot: {
    title: '내적', kind: 'operation', inputs: ['a', 'b'], description: '같은 길이의 두 벡터의 내적을 구한다.', formula: () => 'a·b = Σaᵢbᵢ', compute: (n, [a, b]) => dotValues(a, b)
  },
  derivative: {
    title: '미분', kind: 'calculus', inputs: ['식'], description: '연결된 식을 지정한 변수에 대해 미분한다. 숫자 출력 식의 기울기를 역방향으로 계산한다.',
    formula: n => `∂(식)/∂${n.params.variable}`,
    controls: [{ key: 'variable', label: '미분할 변수 이름', type: 'text', default: 'x' }],
    special: 'derivative'
  },
  display: {
    title: '값 보기', kind: 'sink', inputs: ['x'], description: '계산 결과를 그대로 보여준다.', formula: () => 'x', compute: (n, [x]) => x
  }
};

function op1(title, formula, fn) { return { title, kind: 'operation', inputs: ['x'], description: `${title} 연산을 적용한다.`, formula: () => formula, compute: (n, [x]) => fn(x) }; }
function op2(title, formula, fn) { return { title, kind: 'operation', inputs: ['a', 'b'], description: `${title} 연산을 적용한다. 숫자와 배열의 조합도 지원한다.`, formula: () => formula, compute: (n, [a, b]) => fn(a, b) }; }

function getBlockDef(type) {
  if (BLOCKS[type]) return BLOCKS[type];
  if (type.startsWith('custom:')) {
    const id = type.slice(7), custom = USER_BLOCKS.get(id);
    if (!custom) throw new Error('존재하지 않는 사용자 블록입니다.');
    return {
      title: custom.name, kind: 'custom', inputs: custom.externalInputs.map((x, i) => x.label || `입력 ${i + 1}`),
      description: `${custom.nodes.length}개 블록을 묶어 만든 사용자 블록.`, formula: () => custom.formula || custom.name,
      compute: (node, inputs) => evaluateUserDefinition(custom, inputs)
    };
  }
  throw new Error(`알 수 없는 블록: ${type}`);
}

function addBlock(type, position = null) {
  const def = getBlockDef(type);
  const id = nextNodeId++;
  const count = graph.nodes.size;
  const node = { id, type, x: position?.x ?? 34 + (count % 4) * 205, y: position?.y ?? 34 + Math.floor(count / 4) * 135, params: {} };
  for (const control of def.controls || []) node.params[control.key] = control.default;
  graph.nodes.set(id, node);
  renderNode(node); selectNode(id); syncWorkspaceState(); updateWires();
  return node;
}

function renderNode(node) {
  const def = getBlockDef(node.type);
  const el = document.createElement('div');
  el.className = `node ${def.kind}`; el.dataset.nodeId = node.id; el.style.left = `${node.x}px`; el.style.top = `${node.y}px`;
  const inputsHtml = def.inputs.map((label, index) => `<div class="port-row"><span class="port input" data-node-id="${node.id}" data-port-index="${index}" title="입력 ${label}"></span><span class="port-label">${label}</span></div>`).join('');
  el.innerHTML = `<div class="node-head"><span>${escapeHtml(def.title)}</span><span class="node-kind">${kindLabel(def.kind)}</span></div><div class="node-body">${inputsHtml}${def.inputs.length ? '' : '<span class="muted">입력 없음</span>'}<div class="node-preview">계산 전</div><div class="port-row output-row"><span class="port-label">출력</span><span class="port output" data-node-id="${node.id}" title="출력"></span></div></div>`;
  nodesLayer.appendChild(el);
  installNodeEvents(el, node);
}
function kindLabel(kind) { return ({ source: '입력', data: '데이터', generator: '생성', transform: '변환', operation: '연산', calculus: '미분', custom: '내 블록', sink: '확인' })[kind] || kind; }

function installNodeEvents(el, node) {
  el.querySelector('.port.output').addEventListener('click', event => { event.stopPropagation(); if (!groupSelectionMode) beginConnection(node.id, event.currentTarget); });
  el.querySelectorAll('.port.input').forEach(port => port.addEventListener('click', event => { event.stopPropagation(); if (!groupSelectionMode) finishConnection(node.id, Number(port.dataset.portIndex)); }));
}

nodesLayer.addEventListener('pointerdown', event => {
  const nodeEl = event.target.closest('.node');
  if (!nodeEl || !nodesLayer.contains(nodeEl)) return;
  const nodeId = Number(nodeEl.dataset.nodeId);

  if (groupSelectionMode) {
    event.preventDefault(); event.stopPropagation(); toggleGroupNode(nodeId); return;
  }
  if (event.target.closest('.port')) return;
  event.preventDefault(); event.stopPropagation();
  const node = graph.nodes.get(nodeId); if (!node) return;
  selectNode(nodeId); cancelConnection();
  const start = workspacePoint(event), origin = { x: node.x, y: node.y };
  nodeEl.classList.add('dragging'); nodeEl.setPointerCapture?.(event.pointerId);
  const move = e => { e.preventDefault(); const p = workspacePoint(e); node.x = clamp(origin.x + p.x - start.x, 0, Math.max(0, workspace.clientWidth - nodeEl.offsetWidth)); node.y = clamp(origin.y + p.y - start.y, 0, Math.max(0, workspace.clientHeight - nodeEl.offsetHeight)); nodeEl.style.left = `${node.x}px`; nodeEl.style.top = `${node.y}px`; updateWires(); };
  const finish = e => { nodeEl.classList.remove('dragging'); if (nodeEl.hasPointerCapture?.(e.pointerId)) nodeEl.releasePointerCapture(e.pointerId); nodeEl.removeEventListener('pointermove', move); nodeEl.removeEventListener('pointerup', finish); nodeEl.removeEventListener('pointercancel', finish); };
  nodeEl.addEventListener('pointermove', move); nodeEl.addEventListener('pointerup', finish); nodeEl.addEventListener('pointercancel', finish);
}, true);

function beginConnection(nodeId, portEl) { clearHotPorts(); pendingOutput = nodeId; portEl.classList.add('hot'); document.querySelectorAll('.port.input').forEach(p => p.classList.add('hot')); connectionHint.textContent = '연결할 블록의 왼쪽 입력 ○를 누르세요.'; }
function finishConnection(toNodeId, inputIndex) { if (pendingOutput == null) return; if (pendingOutput === toNodeId) return cancelConnection(); graph.connections = graph.connections.filter(c => !(c.to === toNodeId && c.inputIndex === inputIndex)); graph.connections.push({ from: pendingOutput, to: toNodeId, inputIndex }); cancelConnection(); updateWires(); invalidatePreviews(); }
function cancelConnection() { pendingOutput = null; clearHotPorts(); if (!groupSelectionMode) connectionHint.textContent = '블록을 놓고 수학식을 직접 만드세요.'; }
function clearHotPorts() { document.querySelectorAll('.port.hot').forEach(el => el.classList.remove('hot')); }

function deleteSelectedNode() {
  if (selectedNodeId == null) return;
  removeNodes(new Set([selectedNodeId])); selectedNodeId = null; renderInspector(); syncWorkspaceState(); updateWires();
}
function removeNodes(ids) { for (const id of ids) { graph.nodes.delete(id); nodesLayer.querySelector(`[data-node-id="${id}"]`)?.remove(); } graph.connections = graph.connections.filter(c => !ids.has(c.from) && !ids.has(c.to)); }
function selectNode(id) { selectedNodeId = id; nodesLayer.querySelectorAll('.node').forEach(el => el.classList.toggle('selected', Number(el.dataset.nodeId) === id)); renderInspector(); }

function renderInspector() {
  const node = selectedNodeId == null ? null : graph.nodes.get(selectedNodeId);
  inspectorEmpty.hidden = Boolean(node); inspectorContent.hidden = !node; deleteNodeBtn.disabled = !node;
  if (!node) return;
  const def = getBlockDef(node.type); inspectorTitle.textContent = def.title; inspectorDescription.textContent = def.description; inspectorFormula.textContent = def.formula(node); inspectorValue.textContent = node.lastError ? `오류: ${node.lastError}` : formatValue(node.lastValue); inspectorControls.replaceChildren();
  for (const control of def.controls || []) {
    const row = document.createElement('div'); row.className = 'control-row'; const label = document.createElement('label'); label.textContent = control.label;
    let input;
    if (control.type === 'select') { input = document.createElement('select'); for (const option of control.options || []) { const el = document.createElement('option'); el.value = option.value; el.textContent = option.label; input.appendChild(el); } input.value = node.params[control.key]; }
    else { input = document.createElement('input'); input.type = control.type || 'text'; if (control.step) input.step = control.step; input.value = node.params[control.key]; }
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
    row.append(label, input); inspectorControls.appendChild(row);
  }
}
