// General-purpose state, loop, and matrix tools used to build iterative math systems.

const RUNTIME_VARIABLES = new Map();
let pendingVariableUpdates = null;

function variableSignature(node) {
  return JSON.stringify({
    mode: node.params.mode,
    value: node.params.value,
    length: node.params.length,
    rows: node.params.rows,
    cols: node.params.cols,
    init: node.params.init,
    seed: node.params.seed,
    scale: node.params.scale
  });
}

function randomArray(shape, seed, scale = 1) {
  const length = shape.reduce((a, b) => a * b, 1);
  let state = (Number(seed) || 1) >>> 0;
  const out = new Float32Array(length);
  const s = Number(scale);
  for (let i = 0; i < length; i++) {
    state = (1664525 * state + 1013904223) >>> 0;
    out[i] = ((state / 4294967296) * 2 - 1) * s;
  }
  return arrayValue(out, shape);
}

function initialVariableValue(node) {
  const mode = node.params.mode || 'scalar';
  if (mode === 'scalar') return Number(node.params.value);

  if (mode === 'matrix') {
    const rows = Math.max(1, Math.floor(Number(node.params.rows) || 1));
    const cols = Math.max(1, Math.floor(Number(node.params.cols) || 1));
    if (node.params.init === 'random') return randomArray([rows, cols], node.params.seed, node.params.scale ?? 0.05);
    const data = new Float32Array(rows * cols);
    data.fill(Number(node.params.value));
    return arrayValue(data, [rows, cols]);
  }

  const length = Math.max(1, Math.floor(Number(node.params.length) || 1));
  if (node.params.init === 'random') return randomArray([length], node.params.seed, node.params.scale ?? 0.05);
  const data = new Float32Array(length);
  data.fill(Number(node.params.value));
  return arrayValue(data, [length]);
}

// Runtime variable values are shared, not cloned. Values are immutable by rule
// (see README), so reading a weight matrix hands back the stored array instead
// of copying it on every access inside the training loop.
function readRuntimeVariable(node) {
  const name = String(node.params.name || 'x');
  const signature = variableSignature(node);
  const stored = RUNTIME_VARIABLES.get(name);
  if (!stored || stored.signature !== signature) {
    const value = initialVariableValue(node);
    RUNTIME_VARIABLES.set(name, { value, signature });
    return value;
  }
  return stored.value;
}

function findVariableNode(name) {
  for (const node of graph.nodes.values()) {
    if (node.type === 'variable' && String(node.params.name || 'x') === String(name)) return node;
  }
  return null;
}

function writeRuntimeVariable(name, value, immediate = false) {
  const variableNode = findVariableNode(name);
  if (!variableNode) throw new Error(`'${name}'이라는 변수 블록을 찾지 못했습니다.`);
  const entry = { value, signature: variableSignature(variableNode) };
  if (pendingVariableUpdates && !immediate) pendingVariableUpdates.set(String(name), entry);
  else RUNTIME_VARIABLES.set(String(name), entry);
  return value;
}

function commitPendingVariableUpdates() {
  if (!pendingVariableUpdates) return;
  for (const [name, entry] of pendingVariableUpdates) RUNTIME_VARIABLES.set(name, entry);
}

function matrixVectorValues(matrix, vector) {
  const m = asArrayValue(matrix), v = asArrayValue(vector);
  if (m.shape.length !== 2) throw new Error('첫 번째 입력은 행렬이어야 합니다.');
  if (v.shape.length !== 1) throw new Error('두 번째 입력은 벡터여야 합니다.');
  const [rows, cols] = m.shape;
  if (v.data.length !== cols) throw new Error(`행렬의 열 ${cols}개와 벡터 길이 ${v.data.length}가 다릅니다.`);
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    const base = r * cols;
    for (let c = 0; c < cols; c++) sum += m.data[base + c] * v.data[c];
    out[r] = sum;
  }
  return arrayValue(out, [rows]);
}

function outerProductValues(a, b) {
  const aa = asArrayValue(a), bb = asArrayValue(b);
  if (aa.shape.length !== 1 || bb.shape.length !== 1) throw new Error('외적에는 벡터 두 개가 필요합니다.');
  const out = new Float32Array(aa.data.length * bb.data.length);
  for (let r = 0; r < aa.data.length; r++) {
    for (let c = 0; c < bb.data.length; c++) out[r * bb.data.length + c] = aa.data[r] * bb.data[c];
  }
  return arrayValue(out, [aa.data.length, bb.data.length]);
}

function transposeMatrixVectorValues(matrix, vector) {
  const m = asArrayValue(matrix), v = asArrayValue(vector);
  if (m.shape.length !== 2 || v.shape.length !== 1) throw new Error('전치 행렬 계산의 모양이 맞지 않습니다.');
  const [rows, cols] = m.shape;
  if (v.data.length !== rows) throw new Error('전치 행렬과 벡터의 크기가 맞지 않습니다.');
  const out = new Float32Array(cols);
  for (let c = 0; c < cols; c++) {
    let sum = 0;
    for (let r = 0; r < rows; r++) sum += m.data[r * cols + c] * v.data[r];
    out[c] = sum;
  }
  return arrayValue(out, [cols]);
}

// Extend variables to matrices and make them stateful.
const variableModeControl = BLOCKS.variable.controls.find(control => control.key === 'mode');
if (variableModeControl && !variableModeControl.options.some(option => option.value === 'matrix')) {
  variableModeControl.options.push({ value: 'matrix', label: '행렬' });
}
BLOCKS.variable.controls.push(
  { key: 'rows', label: '행렬 행 수', type: 'number', step: '1', default: 3 },
  { key: 'cols', label: '행렬 열 수', type: 'number', step: '1', default: 3 },
  { key: 'scale', label: '무작위 범위', type: 'number', step: 'any', default: 0.05 }
);
BLOCKS.variable.formula = node => {
  if (node.params.mode === 'matrix') return `${node.params.name} ∈ R^(${node.params.rows}×${node.params.cols})`;
  if (node.params.mode === 'vector') return `${node.params.name} ∈ R^${node.params.length}`;
  return `${node.params.name} = ${node.params.value}`;
};
BLOCKS.variable.compute = node => readRuntimeVariable(node);

BLOCKS.matrix = {
  title: '행렬 만들기', kind: 'generator', inputs: [], description: '원하는 크기의 상수 또는 무작위 행렬을 만든다.',
  formula: node => `A ∈ R^(${node.params.rows}×${node.params.cols})`,
  controls: [
    { key: 'rows', label: '행 수', type: 'number', step: '1', default: 3 },
    { key: 'cols', label: '열 수', type: 'number', step: '1', default: 3 },
    { key: 'init', label: '초기값', type: 'select', default: 'random', options: [{ value: 'constant', label: '같은 값' }, { value: 'random', label: '무작위' }] },
    { key: 'value', label: '같은 값', type: 'number', step: 'any', default: 0 },
    { key: 'seed', label: '무작위 시드', type: 'number', step: '1', default: 1 },
    { key: 'scale', label: '무작위 범위', type: 'number', step: 'any', default: 0.05 }
  ],
  compute: node => {
    const rows = Math.max(1, Math.floor(Number(node.params.rows) || 1));
    const cols = Math.max(1, Math.floor(Number(node.params.cols) || 1));
    if (node.params.init === 'random') return randomArray([rows, cols], node.params.seed, node.params.scale);
    const data = new Float32Array(rows * cols); data.fill(Number(node.params.value)); return arrayValue(data, [rows, cols]);
  }
};

BLOCKS.matvec = {
  title: '행렬 × 벡터', kind: 'operation', inputs: ['행렬', '벡터'], description: '행렬과 벡터를 곱한다. 층 전체의 가중합도 이 연산 하나로 표현할 수 있다.',
  formula: () => 'Ax', compute: (node, [matrix, vector]) => matrixVectorValues(matrix, vector)
};

BLOCKS.setVariable = {
  title: '값 바꾸기', kind: 'operation', inputs: ['새 값'], description: '지정한 변수의 값을 새 값으로 바꾼다. 반복 안에서는 한 회차가 끝날 때 동시에 반영된다.',
  formula: node => `${node.params.variable} ← 새 값`,
  controls: [{ key: 'variable', label: '바꿀 변수 이름', type: 'text', default: 'w' }],
  special: 'setVariable',
  compute: (node, [value]) => writeRuntimeVariable(node.params.variable, value)
};

BLOCKS.sequence = {
  title: '둘 다 계산', kind: 'operation', inputs: ['먼저', '다음'], description: '두 입력을 모두 계산하고 두 번째 결과를 내보낸다. 여러 값 변경을 한 반복 안에 묶을 때 사용한다.',
  formula: () => '먼저 ; 다음', compute: (node, [first, second]) => second
};

BLOCKS.repeat = {
  title: '반복', kind: 'operation', inputs: ['실행할 식'], description: '연결된 계산을 여러 번 다시 실행한다. 반복 번호는 지정한 변수에 0부터 들어간다.',
  formula: node => `repeat ${node.params.count} times (${node.params.indexVariable}=0,1,…)`,
  controls: [
    { key: 'count', label: '반복 횟수', type: 'number', step: '1', default: 10 },
    { key: 'indexVariable', label: '반복 번호 변수', type: 'text', default: 'i' }
  ],
  special: 'repeat'
};

BLOCKS.datasetSampleByIndex = {
  title: '데이터 샘플', kind: 'data', inputs: ['데이터', '번호'], description: '불러온 모든 종류를 번갈아가며 번호에 해당하는 샘플 하나를 꺼낸다.',
  formula: () => 'sample(D, i)',
  compute: (node, [dataset, indexValue]) => {
    if (!dataset || dataset.kind !== 'dataset') throw new Error('불러온 데이터 블록이 필요합니다.');
    if (typeof indexValue !== 'number') throw new Error('샘플 번호는 숫자여야 합니다.');
    const api = window.quickDrawDataset;
    const classes = dataset.classes;
    if (!classes.length) throw new Error('데이터 종류가 없습니다.');
    const i = Math.max(0, Math.floor(indexValue));
    const classIndex = i % classes.length;
    const sampleIndex = Math.floor(i / classes.length) % 10000;
    const className = classes[classIndex];
    const image = api.getNormalizedImage(className, sampleIndex);
    return { kind: 'sample', className, classIndex, classes: [...classes], index: sampleIndex, image };
  }
};

// Add the new blocks to the palette before boot installs click listeners.
(function installTrainingPalette() {
  const sidebar = document.querySelector('.palette-panel');
  const myBlocks = document.querySelector('.my-blocks-group');
  if (!sidebar || !myBlocks) return;
  const section = document.createElement('section');
  section.className = 'palette-group';
  section.innerHTML = `
    <h3>행렬 · 실행</h3>
    <button class="palette-block generator" data-block="matrix">행렬 만들기</button>
    <button class="palette-block operation" data-block="matvec">행렬 × 벡터</button>
    <button class="palette-block operation" data-block="setVariable">값 바꾸기</button>
    <button class="palette-block operation" data-block="sequence">둘 다 계산</button>
    <button class="palette-block operation" data-block="repeat">반복</button>
    <button class="palette-block data" data-block="datasetSampleByIndex">데이터 샘플</button>`;
  sidebar.insertBefore(section, myBlocks);
})();

// Extend automatic differentiation through matrix-vector multiplication.
const primitiveVJPBeforeTraining = primitiveVJP;
primitiveVJP = function(type, inputs, output, upstream) {
  if (type === 'matvec') {
    const [matrix, vector] = inputs;
    const up = asArrayValue(upstream);
    return [outerProductValues(up, asArrayValue(vector)), transposeMatrixVectorValues(matrix, up)];
  }
  if (type === 'sequence') return [zerosLike(inputs[0]), upstream];
  return primitiveVJPBeforeTraining(type, inputs, output, upstream);
};

// Extend evaluation with state changes and real repeated execution.
const evaluateNodeBeforeTraining = evaluateNode;
evaluateNode = function(nodeId, memo = new Map(), visiting = new Set()) {
  if (memo.has(nodeId)) return memo.get(nodeId);
  const node = graph.nodes.get(nodeId);
  if (!node) throw new Error('존재하지 않는 블록입니다.');
  const def = getBlockDef(node.type);

  if (def.special === 'setVariable') {
    if (visiting.has(nodeId)) throw new Error('순환 연결은 계산할 수 없습니다.');
    visiting.add(nodeId);
    const connection = graphInputConnection(nodeId, 0);
    if (!connection) throw new Error("입력 '새 값'이 연결되지 않았습니다.");
    const value = evaluateNode(connection.from, memo, visiting);
    const result = writeRuntimeVariable(String(node.params.variable || 'w'), value);
    memo.set(nodeId, result); visiting.delete(nodeId); return result;
  }

  if (def.special === 'repeat') {
    const connection = graph.connections.find(c => c.to === nodeId && c.inputIndex === 0);
    if (!connection) throw new Error("입력 '실행할 식'이 연결되지 않았습니다.");
    const count = Math.max(0, Math.min(10000, Math.floor(Number(node.params.count) || 0)));
    const indexName = String(node.params.indexVariable || 'i');
    let last = 0;
    for (let i = 0; i < count; i++) {
      const indexNode = findVariableNode(indexName);
      if (indexNode) writeRuntimeVariable(indexName, i, true);
      pendingVariableUpdates = new Map();
      try {
        last = evaluateNode(connection.from, new Map(), new Set());
        commitPendingVariableUpdates();
      } finally {
        pendingVariableUpdates = null;
      }
    }
    memo.set(nodeId, last);
    return last;
  }

  return evaluateNodeBeforeTraining(nodeId, memo, visiting);
};

function collectStateNodesUnderRepeat() {
  const skip = new Set();
  const visit = id => {
    const node = graph.nodes.get(id); if (!node) return;
    const def = getBlockDef(node.type);
    if (def.special === 'setVariable') skip.add(id);
    for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
      const c = graph.connections.find(x => x.to === id && x.inputIndex === inputIndex);
      if (c) visit(c.from);
    }
  };
  for (const node of graph.nodes.values()) {
    if (getBlockDef(node.type).special !== 'repeat') continue;
    const c = graph.connections.find(x => x.to === node.id && x.inputIndex === 0);
    if (c) visit(c.from);
  }
  return skip;
}

const evaluateGraphBeforeTraining = evaluateGraph;
evaluateGraph = function() {
  const memo = new Map(), visiting = new Set();
  const skippedStateNodes = collectStateNodesUnderRepeat();
  let successCount = 0;
  for (const node of graph.nodes.values()) { node.lastValue = undefined; node.lastError = null; }
  for (const node of graph.nodes.values()) {
    if (skippedStateNodes.has(node.id)) { updateNodePreview(node); continue; }
    try { node.lastValue = evaluateNode(node.id, memo, visiting); successCount++; }
    catch (error) { node.lastError = error.message; }
    updateNodePreview(node);
  }
  renderInspector();
  workspaceStatus.textContent = `${successCount}/${graph.nodes.size} 계산`;
};

// Resetting the workspace also resets runtime variable state.
const resetWorkspaceBeforeTraining = resetWorkspace;
resetWorkspace = function() {
  RUNTIME_VARIABLES.clear();
  pendingVariableUpdates = null;
  return resetWorkspaceBeforeTraining();
};

// Stateful control blocks should remain outside reusable pure user blocks for now.
const createUserBlockBeforeTraining = createUserBlockFromSelection;
createUserBlockFromSelection = function() {
  for (const id of groupSelectedIds) {
    const node = graph.nodes.get(id);
    if (!node) continue;
    const special = getBlockDef(node.type).special;
    if (special === 'repeat' || special === 'setVariable') {
      alert('반복과 값 바꾸기는 실행 제어 블록이라 현재는 묶기 바깥에 두세요. 계산식 부분은 자유롭게 묶을 수 있습니다.');
      return;
    }
  }
  return createUserBlockBeforeTraining();
};
