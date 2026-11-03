#!/usr/bin/env node
// 발표 단계 파일 생성기.
//
//   node presentation/build.mjs
//
// presentation/steps.json, descriptions.json, *.mmlab 을 다시 씁니다. 파일을 손으로
// 고치는 대신 이 스크립트를 고치고 다시 실행하세요. bench/presentation-content-check.mjs
// 는 여기서 내보내는 buildAll() 의 기대값으로 실제 페이지에서 각 단계를 계산해 검증합니다.
//
// 좌표계: 노드 폭 190px. 열(col) 간격 300px, 행(row) 간격 190px.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FORMAT = 'machine-learning-math-project';
const VERSION = 1;
const COL = 300;
const ROW = 190;

// ---------- 값 직렬화 (js/file-io.js serializeRuntimeValue 와 같은 형식) ----------

function float32Value(values, shape) {
  const data = Float32Array.from(values);
  return { type: 'float32', shape, data: Buffer.from(data.buffer).toString('base64') };
}

// js/values.js 의 deterministicRandomVector / randomArray 와 비트 단위로 같은 난수.
function lcgFloat32(length, seed, scale = 1) {
  let state = (Number(seed) || 1) >>> 0;
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    state = (1664525 * state + 1013904223) >>> 0;
    out[i] = ((state / 4294967296) * 2 - 1) * scale;
  }
  return out;
}

// ---------- 변수 블록 (js/runtime-variables.js variableSignature 와 같은 서명) ----------

const VARIABLE_DEFAULTS = { mode: 'scalar', value: 1, length: 3, init: 'constant', seed: 1, rows: 3, cols: 3, scale: 0.05 };

function variableParams(name, overrides = {}) {
  return { name, ...VARIABLE_DEFAULTS, ...overrides };
}

function variableSignature(p) {
  return JSON.stringify({ mode: p.mode, value: p.value, length: p.length, rows: p.rows, cols: p.cols, init: p.init, seed: p.seed, scale: p.scale });
}

// ---------- 사용자 블록 정의 ----------

// 수동 역전파 정의의 소스: 입력 i, 순전파 출력 y, 위에서 내려온 기울기 g (js/manual-backprop.js 와 같은 id).
const BACKWARD_SOURCE = { input: i => -(i + 1), y: -1000002, g: -1000001 };

// backward(bb) 는 입력마다 기울기를 내는 블록을 만들고 bb.outputs([...]) 로 입력 순서대로 지정합니다.
function defineBackward(inputCount, build) {
  const bb = {
    nodes: [], connections: [], nextId: 1, gradientOutputNodeIds: null,
    ...BACKWARD_SOURCE,
    add(type, params, col, row) {
      const nodeId = this.nextId++;
      this.nodes.push({ id: nodeId, type, params: params ?? {}, x: 60 + Math.round(col * COL), y: 60 + Math.round(row * ROW) });
      return nodeId;
    },
    op(type, inputs, col, row, params = {}) {
      const nodeId = this.add(type, params, col, row);
      inputs.forEach((from, index) => this.connections.push({ from, to: nodeId, inputIndex: index }));
      return nodeId;
    },
    outputs(ids) { this.gradientOutputNodeIds = ids; }
  };
  build(bb);
  if (bb.gradientOutputNodeIds?.length !== inputCount) throw new Error('입력마다 기울기 출력이 하나씩 필요합니다.');
  return {
    version: 1, inputCount, nodes: bb.nodes, connections: bb.connections,
    gradientOutputNodeIds: bb.gradientOutputNodeIds, editorViewport: { x: 0, y: 0, zoom: 0.75 }
  };
}

function defineBlock({ id, name, formula, uses = [], build, backward = null }) {
  const b = {
    nodes: [], connections: [], externalInputs: [], outputNodeId: null, nextId: 1,
    add(type, params, col, row) {
      const nodeId = this.nextId++;
      this.nodes.push({ id: nodeId, type, params: params ?? {}, x: 60 + Math.round(col * COL), y: 60 + Math.round(row * ROW) });
      return nodeId;
    },
    link(from, to, inputIndex = 0) { this.connections.push({ from, to, inputIndex }); return to; },
    input(nodeId, inputIndex, label) { this.externalInputs.push({ nodeId, inputIndex, label }); },
    output(nodeId) { this.outputNodeId = nodeId; }
  };
  build(b);
  if (b.outputNodeId == null) throw new Error(`${id}: 출력 블록이 없습니다.`);
  const definition = { id, name, nodes: b.nodes, connections: b.connections, externalInputs: b.externalInputs, outputNodeId: b.outputNodeId, formula };
  if (backward) definition.manualBackprop = defineBackward(b.externalInputs.length, backward);
  return { id, uses, definition };
}

// 임계값 함수. step(z) = 1 (z > 0), 0 (그 외) = 1 − [max(z, 0) = 0]
const STEP = defineBlock({
  id: 'step_fn', name: '임계값 함수 · step(z)', formula: 'step(z) = [z > 0] = 1 − [max(z,0) = 0]',
  build(b) {
    const zero = b.add('number', { value: 0 }, 0, 1);
    const mx = b.add('maximum', {}, 1, 0); b.link(zero, mx, 1);
    const eq = b.add('equal', {}, 2, 0); b.link(mx, eq, 0); b.link(zero, eq, 1);
    const one = b.add('number', { value: 1 }, 2, 1);
    const out = b.add('subtract', {}, 3, 0); b.link(one, out, 0); b.link(eq, out, 1);
    b.input(mx, 0, 'z'); b.output(out);
  }
});

// 두 입력 가중합. 입력 순서: x₁, x₂, w₁, w₂, b
function weightedSum(b) {
  const m1 = b.add('multiply', {}, 0, 0);
  const m2 = b.add('multiply', {}, 0, 1);
  const s = b.add('add', {}, 1, 0); b.link(m1, s, 0); b.link(m2, s, 1);
  const z = b.add('add', {}, 2, 0); b.link(s, z, 0);
  b.input(m1, 1, 'x₁'); b.input(m2, 1, 'x₂'); b.input(m1, 0, 'w₁'); b.input(m2, 0, 'w₂'); b.input(z, 1, 'b');
  return z;
}

const PERCEPTRON = defineBlock({
  id: 'perceptron', name: '퍼셉트론 · step(w₁x₁ + w₂x₂ + b)', formula: 'step(w₁x₁ + w₂x₂ + b)', uses: [STEP],
  build(b) {
    const z = weightedSum(b);
    const out = b.add('custom:step_fn', {}, 3, 0); b.link(z, out, 0);
    b.output(out);
  }
});

// 역전파: ∂x₁ = g·w₁, ∂x₂ = g·w₂, ∂w₁ = Σ g·x₁, ∂w₂ = Σ g·x₂, ∂b = Σ g
// (w 와 b 는 숫자 하나가 네 경우 모두에 쓰이므로 네 경우의 기울기를 더합니다.)
const LINEAR = defineBlock({
  id: 'linear2', name: '선형 모형 · w₁x₁ + w₂x₂ + b', formula: 'ŷ = w₁x₁ + w₂x₂ + b',
  build(b) { b.output(weightedSum(b)); },
  backward(bb) {
    const [x1, x2, w1, w2] = [0, 1, 2, 3].map(bb.input);
    const dx1 = bb.op('multiply', [bb.g, w1], 0, 0);
    const dx2 = bb.op('multiply', [bb.g, w2], 0, 1);
    const dw1 = bb.op('sum', [bb.op('multiply', [bb.g, x1], 1, 0)], 2, 0);
    const dw2 = bb.op('sum', [bb.op('multiply', [bb.g, x2], 1, 1)], 2, 1);
    const db = bb.op('sum', [bb.g], 2, 2);
    bb.outputs([dx1, dx2, dw1, dw2, db]);
  }
});

function gate(id, name, formula, w1, w2, bias) {
  return defineBlock({
    id, name, formula, uses: [PERCEPTRON],
    build(b) {
      const n1 = b.add('number', { value: w1 }, 0, 0);
      const n2 = b.add('number', { value: w2 }, 0, 1);
      const nb = b.add('number', { value: bias }, 0, 2);
      const p = b.add('custom:perceptron', {}, 1, 0);
      b.link(n1, p, 2); b.link(n2, p, 3); b.link(nb, p, 4);
      b.input(p, 0, 'x₁'); b.input(p, 1, 'x₂'); b.output(p);
    }
  });
}

const GATES = {
  AND: { w: [1, 1, -1.5], block: gate('and_gate', 'AND 게이트 · 퍼셉트론(1, 1, −1.5)', 'step(x₁ + x₂ − 1.5)', 1, 1, -1.5) },
  OR: { w: [1, 1, -0.5], block: gate('or_gate', 'OR 게이트 · 퍼셉트론(1, 1, −0.5)', 'step(x₁ + x₂ − 0.5)', 1, 1, -0.5) },
  NAND: { w: [-1, -1, 1.5], block: gate('nand_gate', 'NAND 게이트 · 퍼셉트론(−1, −1, 1.5)', 'step(−x₁ − x₂ + 1.5)', -1, -1, 1.5) }
};

// 하나의 입력을 여러 곳에 쓰려면 안에서 값 보기 블록으로 갈라 씁니다 (입력 포트 하나 = 슬롯 하나).
const XOR = defineBlock({
  id: 'xor_gate', name: 'XOR 게이트 · AND(OR, NAND)', formula: 'AND(OR(x₁,x₂), NAND(x₁,x₂))',
  uses: [GATES.AND.block, GATES.OR.block, GATES.NAND.block],
  build(b) {
    const px1 = b.add('display', {}, 0, 0), px2 = b.add('display', {}, 0, 1);
    const o = b.add('custom:or_gate', {}, 1, 0), n = b.add('custom:nand_gate', {}, 1, 1);
    b.link(px1, o, 0); b.link(px2, o, 1); b.link(px1, n, 0); b.link(px2, n, 1);
    const a = b.add('custom:and_gate', {}, 2, 0); b.link(o, a, 0); b.link(n, a, 1);
    b.input(px1, 0, 'x₁'); b.input(px2, 0, 'x₂'); b.output(a);
  }
});

// 역전파: ∂ŷ = g·2(ŷ − y), ∂y = −∂ŷ
const SQ_ERR = defineBlock({
  id: 'sq_err', name: '손실 · Σ(ŷ − y)²', formula: 'Σ(ŷ − y)²',
  build(b) {
    const e = b.add('subtract', {}, 0, 0);
    const sq = b.add('multiply', {}, 1, 0); b.link(e, sq, 0); b.link(e, sq, 1);
    const out = b.add('sum', {}, 2, 0); b.link(sq, out, 0);
    b.input(e, 0, 'ŷ'); b.input(e, 1, 'y'); b.output(out);
  },
  backward(bb) {
    const two = bb.add('number', { value: 2 }, 0, 1), zero = bb.add('number', { value: 0 }, 2, 1);
    const e = bb.op('subtract', [bb.input(0), bb.input(1)], 0, 0);
    const dyhat = bb.op('multiply', [bb.g, bb.op('multiply', [two, e], 1, 0)], 2, 0);
    bb.outputs([dyhat, bb.op('subtract', [zero, dyhat], 3, 1)]);
  }
});

const SGD_STEP = defineBlock({
  id: 'sgd_step', name: '경사 한 걸음 · p − η·g', formula: 'p − η·g',
  build(b) {
    const m = b.add('multiply', {}, 0, 1);
    const out = b.add('subtract', {}, 1, 0); b.link(m, out, 1);
    b.input(out, 0, 'p'); b.input(m, 0, 'η'); b.input(m, 1, 'g'); b.output(out);
  }
});

const PICK = defineBlock({
  id: 'pick', name: '원소 꺼내기 · w[j]', formula: 'w · [idx = j]',
  build(b) {
    const eq = b.add('equal', {}, 0, 1);
    const out = b.add('dot', {}, 1, 0); b.link(eq, out, 1);
    b.input(out, 0, 'w'); b.input(eq, 0, 'idx'); b.input(eq, 1, 'j'); b.output(out);
  }
});

// 파라미터 9개를 벡터 하나로 받는 2-2-1 임계값 신경망.
// h₁ = step(w₀x₁ + w₁x₂ + w₂), h₂ = step(w₃x₁ + w₄x₂ + w₅), ŷ = step(w₆h₁ + w₇h₂ + w₈)
const NET9 = defineBlock({
  id: 'net9', name: '2-2-1 임계값 신경망 · w ∈ R⁹', formula: 'ŷ = step(w₆h₁ + w₇h₂ + w₈),  hᵢ = step(w·x + b)',
  uses: [PICK, PERCEPTRON],
  build(b) {
    const px1 = b.add('display', {}, 0, 0), px2 = b.add('display', {}, 0, 1);
    const pw = b.add('display', {}, 0, 2), pidx = b.add('display', {}, 0, 3);
    const c = [];
    for (let j = 0; j < 9; j++) {
      const nj = b.add('number', { value: j }, 1, j);
      const pick = b.add('custom:pick', {}, 2, j);
      b.link(pw, pick, 0); b.link(pidx, pick, 1); b.link(nj, pick, 2);
      c.push(pick);
    }
    const h1 = b.add('custom:perceptron', {}, 3, 0);
    b.link(px1, h1, 0); b.link(px2, h1, 1); b.link(c[0], h1, 2); b.link(c[1], h1, 3); b.link(c[2], h1, 4);
    const h2 = b.add('custom:perceptron', {}, 3, 3);
    b.link(px1, h2, 0); b.link(px2, h2, 1); b.link(c[3], h2, 2); b.link(c[4], h2, 3); b.link(c[5], h2, 4);
    const out = b.add('custom:perceptron', {}, 4, 6);
    b.link(h1, out, 0); b.link(h2, out, 1); b.link(c[6], out, 2); b.link(c[7], out, 3); b.link(c[8], out, 4);
    b.input(px1, 0, 'x₁'); b.input(px2, 0, 'x₂'); b.input(pw, 0, 'w (R⁹)'); b.input(pidx, 0, 'idx = [0…8]');
    b.output(out);
  }
});

const SELECT = defineBlock({
  id: 'select', name: '채택 · s = 1이면 후보 c, 0이면 w 유지', formula: 'w + s·(c − w)',
  build(b) {
    const pw = b.add('display', {}, 0, 0);
    const d = b.add('subtract', {}, 1, 1); b.link(pw, d, 1);
    const m = b.add('multiply', {}, 2, 1); b.link(d, m, 1);
    const out = b.add('add', {}, 3, 0); b.link(pw, out, 0); b.link(m, out, 1);
    b.input(pw, 0, 'w'); b.input(d, 0, 'c'); b.input(m, 0, 's'); b.output(out);
  }
});

const RELU = defineBlock({
  id: 'relu', name: 'ReLU · max(x, 0)', formula: 'max(x, 0)',
  build(b) {
    const zero = b.add('number', { value: 0 }, 0, 1);
    const out = b.add('maximum', {}, 1, 0); b.link(zero, out, 1);
    b.input(out, 0, 'x'); b.output(out);
  }
});

const SQUARE = defineBlock({
  id: 'square', name: '제곱 · x²', formula: 'x · x',
  build(b) {
    const px = b.add('display', {}, 0, 0);
    const out = b.add('multiply', {}, 1, 0); b.link(px, out, 0); b.link(px, out, 1);
    b.input(px, 0, 'x'); b.output(out);
  }
});

const SQ_RELU = defineBlock({
  id: 'sq_relu', name: '블록 안의 블록 · ReLU(x)²', formula: 'max(x, 0)²', uses: [RELU, SQUARE],
  build(b) {
    const r = b.add('custom:relu', {}, 0, 0);
    const s = b.add('custom:square', {}, 1, 0); b.link(r, s, 0);
    b.input(r, 0, 'x'); b.output(s);
  }
});

// ---------- 작업공간 ----------

class Workspace {
  constructor() {
    this.nodes = []; this.connections = []; this.userBlocks = new Map(); this.runtimeVariables = [];
    this.nextId = 1; this.checks = [];
  }
  add(type, params, col, row) {
    const id = this.nextId++;
    this.nodes.push({ id, type, x: 40 + Math.round(col * COL), y: 40 + Math.round(row * ROW), params: params ?? {} });
    return id;
  }
  link(from, to, inputIndex = 0) { this.connections.push({ from, to, inputIndex }); return to; }
  number(value, col, row) { return this.add('number', { value }, col, row); }
  // seedValues 를 주면 그 값을 저장된 런타임 값으로 함께 내보냅니다 (진리표, 색인 벡터 등).
  variable(name, overrides, col, row, seedValues = null) {
    const params = variableParams(name, overrides);
    const id = this.add('variable', params, col, row);
    if (seedValues) {
      const shape = params.mode === 'matrix' ? [params.rows, params.cols] : [params.length];
      this.runtimeVariables.push({ name, signature: variableSignature(params), value: float32Value(seedValues, shape) });
    }
    return id;
  }
  display(from, col, row, check = null, params = {}) {
    const id = this.add('display', params, col, row);
    this.link(from, id, 0);
    if (check) this.checks.push({ id, ...check });
    return id;
  }
  // 사용자 블록을 의존 블록까지 포함해 등록합니다.
  use(...blocks) {
    for (const block of blocks) {
      if (this.userBlocks.has(block.id)) continue;
      this.use(...block.uses);
      this.userBlocks.set(block.id, JSON.parse(JSON.stringify(block.definition)));
    }
    return this;
  }
  custom(block, inputs, col, row, params = {}) {
    this.use(block);
    const id = this.add(`custom:${block.id}`, params, col, row);
    inputs.forEach((from, index) => this.link(from, id, index));
    return id;
  }
  op(type, inputs, col, row, params = {}) {
    const id = this.add(type, params, col, row);
    inputs.forEach((from, index) => this.link(from, id, index));
    return id;
  }
  // 둘 다 계산 블록을 사슬로 엮어 여러 값 바꾸기를 한 회차에 묶습니다.
  chain(ids, col, row) {
    let body = ids[0];
    ids.slice(1).forEach((id, k) => { body = this.op('sequence', [body, id], col + k, row); });
    return body;
  }
  // builtins: 기본 블록(곱하기 등)에 붙인 역전파 정의 [{ type, definition }].
  toProject(viewport, datasetSelection = [], builtins = null) {
    const project = {
      format: FORMAT, version: VERSION,
      graph: { nodes: this.nodes, connections: this.connections },
      userBlocks: Array.from(this.userBlocks.values()),
      runtimeVariables: this.runtimeVariables,
      datasetSelection,
      viewport
    };
    if (builtins) project.manualBackprop = { version: 1, builtins };
    return project;
  }
}

// 진리표: 네 경우를 길이 4 벡터로 한 번에 담습니다.
const TRUTH = { x1: [0, 0, 1, 1], x2: [0, 1, 0, 1], xor: [0, 1, 1, 0] };

function truthInputs(ws, col = 0, row = 0, withDisplays = true) {
  const x1 = ws.variable('x1', { mode: 'vector', length: 4, value: 0 }, col, row, TRUTH.x1);
  const x2 = ws.variable('x2', { mode: 'vector', length: 4, value: 0 }, col, row + 1, TRUTH.x2);
  if (withDisplays) {
    ws.display(x1, col + 1, row, { expect: TRUTH.x1 });
    ws.display(x2, col + 1, row + 1, { expect: TRUTH.x2 });
  }
  return { x1, x2 };
}

// ---------- 0단계: 블록 사용법 ----------

function build00a() {
  const ws = new Workspace();
  ws.display(ws.number(3, 0, 0), 1, 0, { expect: 3 });
  ws.display(ws.variable('a', { value: 2 }, 0, 1), 1, 1, { expect: 2 });
  ws.display(ws.variable('v', { mode: 'vector', length: 3, value: 0.5 }, 0, 2), 1, 2, { expect: [0.5, 0.5, 0.5] });
  ws.display(ws.variable('r', { mode: 'vector', length: 5, init: 'random', seed: 7, scale: 1 }, 0, 3), 1, 3, { expect: Array.from(lcgFloat32(5, 7, 1)) });
  ws.display(ws.variable('M', { mode: 'matrix', rows: 2, cols: 3, init: 'random', seed: 3, scale: 1 }, 0, 4), 1, 4, { expect: Array.from(lcgFloat32(6, 3, 1)), shape: [2, 3] });
  ws.display(ws.add('constantVector', { length: 4, value: 1 }, 3, 0), 4, 0, { expect: [1, 1, 1, 1] });
  ws.display(ws.add('randomVector', { length: 4, seed: 2 }, 3, 1), 4, 1, { expect: Array.from(lcgFloat32(4, 2, 1)) });
  ws.display(ws.add('matrix', { rows: 2, cols: 2, init: 'constant', value: 0, seed: 1, scale: 0.05 }, 3, 2), 4, 2, { expect: [0, 0, 0, 0], shape: [2, 2] });
  ws.display(ws.add('matrix', { rows: 3, cols: 2, init: 'random', value: 0, seed: 5, scale: 1 }, 3, 3), 4, 3, { expect: Array.from(lcgFloat32(6, 5, 1)), shape: [3, 2] });
  return { ws, viewport: { x: 0, y: 0, zoom: 0.8 } };
}

function build00b() {
  const ws = new Workspace();
  // 예시 하나가 세로 1.75행을 차지합니다 (블록 높이 133px > 행 간격의 절반).
  const two = (type, a, b, example, col0, expect) => {
    const row = example * 1.75;
    const na = typeof a === 'number' ? ws.number(a, col0, row) : a(col0, row);
    const nb = typeof b === 'number' ? ws.number(b, col0, row + 0.75) : b(col0, row + 0.75);
    ws.display(ws.op(type, [na, nb], col0 + 1, row), col0 + 2, row, { expect });
  };
  const cvec = (length, value) => (col, row) => ws.add('constantVector', { length, value }, col, row);
  const rvec = (length, seed) => (col, row) => ws.add('randomVector', { length, seed }, col, row);
  const r5 = Array.from(lcgFloat32(5, 2, 1));

  // 왼쪽 열
  two('add', 2, 3, 0, 0, 5);
  two('subtract', 5, 2, 1, 0, 3);
  two('multiply', 4, 2.5, 2, 0, 10);
  two('divide', 1, 4, 3, 0, 0.25);
  two('add', cvec(3, 1), 2, 4, 0, [3, 3, 3]);
  two('multiply', cvec(3, 2), cvec(3, 4), 5, 0, [8, 8, 8]);
  two('maximum', rvec(5, 2), 0, 6, 0, r5.map(v => Math.max(v, 0)));
  // 오른쪽 열
  two('equal', 2, 3, 0, 4, 0);
  ws.display(ws.op('sum', [cvec(4, 2)(4, 1.75)], 5, 1.75), 6, 1.75, { expect: 8 });
  two('dot', cvec(3, 2), cvec(3, 3), 2, 4, 18);
  two('matvec', (c, r) => ws.add('matrix', { rows: 2, cols: 3, init: 'constant', value: 1, seed: 1, scale: 0.05 }, c, r), cvec(3, 2), 3, 4, [6, 6]);
  ws.display(ws.op('exp', [ws.number(1, 4, 7)], 5, 7), 6, 7, { approx: Math.E, tol: 1e-5 });
  ws.display(ws.op('log', [ws.number(10, 4, 8.75)], 5, 8.75), 6, 8.75, { approx: Math.log(10), tol: 1e-5 });
  ws.display(ws.op('arrayMax', [rvec(5, 2)(4, 10.5)], 5, 10.5), 6, 10.5, { approx: Math.max(...r5), tol: 1e-6 });
  return { ws, viewport: { x: 0, y: 0, zoom: 0.7 } };
}

function build00c() {
  const ws = new Workspace();
  // A. n ← n + 1 을 10번
  const n = ws.variable('n', { value: 0 }, 0, 0);
  const inc = ws.op('add', [n, ws.number(1, 0, 1)], 1, 0);
  const setN = ws.op('setVariable', [inc], 2, 0, { variable: 'n' });
  ws.op('repeat', [setN], 3, 0, { count: 10, start: 0, indexVariable: 'i' });
  ws.display(ws.variable('n', { value: 0 }, 4, 0), 5, 0, { expect: 10 });

  // B. acc ← acc + i² (i = 1…5)
  const i = ws.variable('i', { value: 0 }, 0, 2.5);
  const acc = ws.variable('acc', { value: 0 }, 0, 3.5);
  const sq = ws.op('multiply', [i, i], 1, 2.5);
  const addSq = ws.op('add', [acc, sq], 2, 2.5);
  const setAcc = ws.op('setVariable', [addSq], 3, 2.5, { variable: 'acc' });
  ws.op('repeat', [setAcc], 4, 2.5, { count: 5, start: 1, indexVariable: 'i' });
  ws.display(ws.variable('acc', { value: 0 }, 5, 2.5), 6, 2.5, { expect: 55 });

  // C. a ← b, b ← a + b 를 10번 (피보나치)
  const a = ws.variable('a', { value: 1 }, 0, 5);
  const bVar = ws.variable('b', { value: 1 }, 0, 6);
  const setA = ws.op('setVariable', [bVar], 2, 5, { variable: 'a' });
  const sum = ws.op('add', [a, bVar], 1, 6);
  const setB = ws.op('setVariable', [sum], 2, 6, { variable: 'b' });
  const both = ws.op('sequence', [setA, setB], 3, 5);
  ws.op('repeat', [both], 4, 5, { count: 10, start: 0, indexVariable: 'i' });
  ws.display(ws.variable('a', { value: 1 }, 5, 5), 6, 5, { expect: 89 });
  ws.display(ws.variable('b', { value: 1 }, 5, 6), 6, 6, { expect: 144 });
  return { ws, viewport: { x: 0, y: 0, zoom: 0.7 } };
}

function build00d() {
  const ws = new Workspace();
  const r = Array.from(lcgFloat32(5, 2, 1));
  const input = ws.add('randomVector', { length: 5, seed: 2 }, 0, 0);
  ws.display(input, 1, 0, { expect: r });
  ws.display(ws.custom(RELU, [input], 1, 1), 2, 1, { expect: r.map(v => Math.max(v, 0)) });
  ws.display(ws.custom(SQUARE, [input], 1, 2), 2, 2, { expect: r.map(v => Math.fround(v * v)) });
  ws.display(ws.custom(SQ_RELU, [input], 1, 3), 2, 3, { expect: r.map(v => Math.fround(Math.max(v, 0) ** 2)) });
  // 발표 중 직접 묶어 볼 재료: max(x,0) 두 블록
  const raw = ws.op('maximum', [ws.number(-3, 0, 5), ws.number(0, 0, 6)], 1, 5);
  ws.display(raw, 2, 5, { expect: 0 });
  return { ws, viewport: { x: 0, y: 0, zoom: 0.8 } };
}

// ---------- 1단계: 퍼셉트론 ----------

function stepFn(z) { return z > 0 ? 1 : 0; }
function perceptronTable(w1, w2, b, x1 = TRUTH.x1, x2 = TRUTH.x2) {
  return x1.map((_, k) => stepFn(Math.fround(Math.fround(Math.fround(w1 * x1[k]) + Math.fround(w2 * x2[k])) + b)));
}

function build01() {
  const ws = new Workspace();
  const { x1, x2 } = truthInputs(ws);

  // 직접 조립: 곱하기 → 더하기 → 더하기 → 임계값 함수
  const w1 = ws.number(1, 0, 3), w2 = ws.number(1, 0, 4), b = ws.number(-1.5, 0, 5);
  const m1 = ws.op('multiply', [w1, x1], 1, 3);
  const m2 = ws.op('multiply', [w2, x2], 1, 4);
  const s = ws.op('add', [m1, m2], 2, 3);
  const z = ws.op('add', [s, b], 3, 3);
  ws.display(z, 4, 4, { expect: [-1.5, -0.5, -0.5, 0.5] });
  ws.display(ws.custom(STEP, [z], 4, 3), 5, 3, { expect: [0, 0, 0, 1] });

  // 퍼셉트론 사용자 블록
  const pw1 = ws.number(1, 0, 7), pw2 = ws.number(1, 0, 8), pb = ws.number(-1.5, 0, 9);
  ws.display(ws.custom(PERCEPTRON, [x1, x2, pw1, pw2, pb], 1, 7), 2, 7, { expect: [0, 0, 0, 1] });

  // 게이트 블록
  let row = 11;
  for (const name of ['AND', 'OR', 'NAND']) {
    const gateDef = GATES[name];
    ws.display(ws.custom(gateDef.block, [x1, x2], 1, row), 2, row, { expect: perceptronTable(...gateDef.w) });
    row += 1;
  }
  return { ws, viewport: { x: 0, y: 0, zoom: 0.6 } };
}

// ---------- 2단계: XOR ----------

function build02() {
  const ws = new Workspace();
  const { x1, x2 } = truthInputs(ws);
  const y = ws.variable('y', { mode: 'vector', length: 4, value: 0 }, 0, 2, TRUTH.xor);
  ws.display(y, 1, 2, { expect: TRUTH.xor });

  const o = ws.custom(GATES.OR.block, [x1, x2], 1, 4);
  const n = ws.custom(GATES.NAND.block, [x1, x2], 1, 5);
  const a = ws.custom(GATES.AND.block, [o, n], 2, 4);
  ws.display(a, 3, 4, { expect: TRUTH.xor });

  const xor = ws.custom(XOR, [x1, x2], 1, 7);
  ws.display(xor, 2, 7, { expect: TRUTH.xor });
  const eq = ws.op('equal', [xor, y], 3, 7);
  ws.display(ws.op('sum', [eq], 4, 7), 5, 7, { expect: 4 });
  return { ws, viewport: { x: 0, y: 0, zoom: 0.7 } };
}

// ---------- 3단계: 선형 모형 학습 실패 ----------

const LINEAR_INIT = { w1: 0.5, w2: -0.3, b: 0.2 };
const LINEAR_LR = 0.1;
const LINEAR_STEPS = 200;

// 페이지의 float32 계산과 같은 순서로 경사하강을 흉내내 기대값을 만듭니다.
function simulateLinear() {
  const f = Math.fround;
  let { w1, w2, b } = LINEAR_INIT;
  const predict = () => TRUTH.x1.map((_, k) => f(f(f(w1 * TRUTH.x1[k]) + f(w2 * TRUTH.x2[k])) + b));
  const loss = yhat => yhat.reduce((acc, v, k) => f(acc + f(f(v - TRUTH.xor[k]) * f(v - TRUTH.xor[k]))), 0);
  const before = predict();
  for (let t = 0; t < LINEAR_STEPS; t++) {
    const yhat = predict();
    const e = yhat.map((v, k) => f(v - TRUTH.xor[k]));
    const g1 = e.reduce((acc, v, k) => f(acc + f(v * TRUTH.x1[k])), 0);
    const g2 = e.reduce((acc, v, k) => f(acc + f(v * TRUTH.x2[k])), 0);
    const gb = e.reduce((acc, v) => f(acc + v), 0);
    w1 = f(w1 - f(LINEAR_LR * g1)); w2 = f(w2 - f(LINEAR_LR * g2)); b = f(b - f(LINEAR_LR * gb));
  }
  const after = predict();
  return { before, lossBefore: loss(before), after, lossAfter: loss(after), w1, w2, b };
}

function build03() {
  const ws = new Workspace();
  const sim = simulateLinear();
  const { x1, x2 } = truthInputs(ws, 0, 0, false);
  const y = ws.variable('y', { mode: 'vector', length: 4, value: 0 }, 0, 2, TRUTH.xor);
  ws.display(y, 1, 2, { expect: TRUTH.xor });

  const w1 = ws.variable('w1', { value: LINEAR_INIT.w1, trainable: 'yes' }, 0, 4);
  const w2 = ws.variable('w2', { value: LINEAR_INIT.w2, trainable: 'yes' }, 0, 5);
  const b = ws.variable('b', { value: LINEAR_INIT.b, trainable: 'yes' }, 0, 6);
  const eta = ws.number(LINEAR_LR, 0, 7);

  // 학습 전
  const yhat = ws.custom(LINEAR, [x1, x2, w1, w2, b], 2, 4);
  ws.display(yhat, 3, 3, { approx: sim.before, tol: 1e-6 });
  ws.display(ws.custom(SQ_ERR, [yhat, y], 3, 4), 4, 4, { approx: sim.lossBefore, tol: 1e-5 });

  // 오차 → 기울기 (퍼셉트론 학습 규칙과 같은 모양)
  const e = ws.op('subtract', [yhat, y], 3, 6);
  const g1 = ws.op('sum', [ws.op('multiply', [e, x1], 4, 6)], 5, 6);
  const g2 = ws.op('sum', [ws.op('multiply', [e, x2], 4, 7.25)], 5, 7.25);
  const gb = ws.op('sum', [e], 5, 8.5);

  // 경사 한 걸음 → 값 바꾸기 → 반복 (경사 한 걸음 블록은 입력이 3개라 1.25행 간격)
  const updates = [[w1, g1, 'w1', 6], [w2, g2, 'w2', 7.25], [b, gb, 'b', 8.5]].map(([p, g, name, row]) =>
    ws.op('setVariable', [ws.custom(SGD_STEP, [p, eta, g], 6, row)], 7, row, { variable: name }));
  const body = ws.chain(updates, 8, 6);
  ws.op('repeat', [body], 10, 6, { count: LINEAR_STEPS, start: 0, indexVariable: 'i' });

  // 학습 후 (반복보다 나중에 만든 변수 블록이 갱신된 값을 읽습니다)
  const w1After = ws.variable('w1', { value: LINEAR_INIT.w1, trainable: 'yes' }, 0, 11.5);
  const w2After = ws.variable('w2', { value: LINEAR_INIT.w2, trainable: 'yes' }, 0, 12.5);
  const bAfter = ws.variable('b', { value: LINEAR_INIT.b, trainable: 'yes' }, 0, 13.5);
  ws.display(w1After, 1, 11.5, { approx: sim.w1, tol: 1e-4 });
  ws.display(w2After, 1, 12.5, { approx: sim.w2, tol: 1e-4 });
  ws.display(bAfter, 1, 13.5, { approx: sim.b, tol: 1e-4 });
  const yhatAfter = ws.custom(LINEAR, [x1, x2, w1After, w2After, bAfter], 2, 11.5);
  ws.display(yhatAfter, 3, 10.5, { approx: sim.after, tol: 1e-4 });
  ws.display(ws.custom(SQ_ERR, [yhatAfter, y], 3, 11.5), 4, 11.5, { approx: sim.lossAfter, tol: 1e-4 });
  return { ws, viewport: { x: 0, y: 0, zoom: 0.5 }, sim };
}

// ---------- 4단계: 임계값 활성화, 탐색 학습 ----------

const SEARCH_N = 3000;       // 후보 수 = 반복 횟수
const SEARCH_W_SEED = 1;     // 초기 w 시드

function net9Table(w) {
  const h1 = perceptronTable(w[0], w[1], w[2]);
  const h2 = perceptronTable(w[3], w[4], w[5]);
  return perceptronTable(w[6], w[7], w[8], h1, h2);
}
function errorCount(yhat) { return yhat.reduce((acc, v, k) => acc + (v - TRUTH.xor[k]) ** 2, 0); }

// 페이지와 같은 난수, 같은 채택 규칙으로 탐색을 흉내냅니다.
function simulateSearch(candidateSeed) {
  const A = lcgFloat32(9 * SEARCH_N, candidateSeed, 1); // 9행 × N열, 행 우선
  let w = Array.from(lcgFloat32(9, SEARCH_W_SEED, 1));
  const initialError = errorCount(net9Table(w));
  let tries = 0;
  for (let i = 0; i < SEARCH_N; i++) {
    const c = Array.from({ length: 9 }, (_, r) => A[r * SEARCH_N + i]);
    const eCur = errorCount(net9Table(w));
    const eCand = errorCount(net9Table(c));
    if (eCur > 0) tries++;
    if (eCand <= eCur) w = c;
  }
  return { w, initialError, finalError: errorCount(net9Table(w)), tries };
}

// 발표에서 보여 줄 만한 시도 횟수(수백~천여 번)가 나오는 후보 시드를 고릅니다.
function chooseSearchSeed() {
  for (let seed = 1; seed < 500; seed++) {
    const result = simulateSearch(seed);
    if (result.finalError === 0 && result.tries >= 600 && result.tries <= 1400) return { seed, ...result };
  }
  throw new Error('조건에 맞는 탐색 시드를 찾지 못했습니다.');
}

function build04() {
  const ws = new Workspace();
  const search = chooseSearchSeed();
  const { x1, x2 } = truthInputs(ws);

  // 위: 손으로 넣은 가중치로 2-2-1 신경망 = 2단계의 XOR 회로
  const numbers = (values, col, row) => values.map((v, k) => ws.number(v, col, row + k * 0.7));
  const h1 = ws.custom(PERCEPTRON, [x1, x2, ...numbers(GATES.OR.w, 2, 0)], 3, 0);
  const h2 = ws.custom(PERCEPTRON, [x1, x2, ...numbers(GATES.NAND.w, 2, 2.2)], 3, 2.2);
  const out = ws.custom(PERCEPTRON, [h1, h2, ...numbers(GATES.AND.w, 4, 0.5)], 5, 1);
  ws.display(out, 6, 1, { expect: TRUTH.xor });

  // 아래: 탐색 학습
  const R = 5.5;
  const y = ws.variable('y', { mode: 'vector', length: 4, value: 0 }, 0, R, TRUTH.xor);
  ws.display(y, 1, R, { expect: TRUTH.xor });
  const idx9 = ws.variable('idx9', { mode: 'vector', length: 9, value: 0 }, 0, R + 1, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
  const idxN = ws.variable('idxN', { mode: 'vector', length: SEARCH_N, value: 0 }, 0, R + 2, Array.from({ length: SEARCH_N }, (_, k) => k));
  const i = ws.variable('i', { value: 0 }, 0, R + 3);
  const w = ws.variable('w', { mode: 'vector', length: 9, init: 'random', seed: SEARCH_W_SEED, scale: 1, trainable: 'yes' }, 0, R + 4);
  ws.display(w, 1, R + 4, { expect: Array.from(lcgFloat32(9, SEARCH_W_SEED, 1)) });
  const tries = ws.variable('tries', { value: 0 }, 0, R + 5);
  const candidates = ws.add('matrix', { rows: 9, cols: SEARCH_N, init: 'random', value: 0, seed: search.seed, scale: 1 }, 0, R + 6);

  // i번째 후보 c = A · [idxN = i]
  const onehot = ws.op('equal', [idxN, i], 2, R + 2);
  const c = ws.op('matvec', [candidates, onehot], 3, R + 2);

  // 현재 w 와 후보 c 의 오답 개수
  const yCur = ws.custom(NET9, [x1, x2, w, idx9], 2, R + 4);
  ws.display(yCur, 3, R + 5, { expect: net9Table(Array.from(lcgFloat32(9, SEARCH_W_SEED, 1))) });
  const eCur = ws.custom(SQ_ERR, [yCur, y], 3, R + 4);
  ws.display(eCur, 4, R + 5, { expect: search.initialError });
  const yCand = ws.custom(NET9, [x1, x2, c, idx9], 4, R + 3);
  const eCand = ws.custom(SQ_ERR, [yCand, y], 5, R + 3);

  // 채택 여부 s = [max(e_c, e_w) = e_w] (후보가 더 나쁘지 않으면 1)
  const worst = ws.op('maximum', [eCand, eCur], 6, R + 3);
  const s = ws.op('equal', [worst, eCur], 7, R + 3);
  const setW = ws.op('setVariable', [ws.custom(SELECT, [w, c, s], 8, R + 3)], 9, R + 3, { variable: 'w' });

  // 아직 못 맞춘 회차 수: tries ← tries + (1 − [e_w = 0])
  const zero = ws.number(0, 5, R + 5.5), one = ws.number(1, 6, R + 5.5);
  const solved = ws.op('equal', [eCur, zero], 6, R + 4.6);
  const notYet = ws.op('subtract', [one, solved], 7, R + 4.6);
  const setTries = ws.op('setVariable', [ws.op('add', [tries, notYet], 8, R + 4.6)], 9, R + 4.6, { variable: 'tries' });

  const body = ws.op('sequence', [setW, setTries], 10, R + 3);
  ws.op('repeat', [body], 11, R + 3, { count: SEARCH_N, start: 0, indexVariable: 'i' });

  // 탐색 후
  const R2 = R + 8.5;
  const wAfter = ws.variable('w', { mode: 'vector', length: 9, init: 'random', seed: SEARCH_W_SEED, scale: 1, trainable: 'yes' }, 0, R2);
  ws.display(wAfter, 1, R2, { approx: search.w, tol: 1e-6 });
  const yAfter = ws.custom(NET9, [x1, x2, wAfter, idx9], 2, R2);
  ws.display(yAfter, 3, R2, { expect: TRUTH.xor });
  ws.display(ws.custom(SQ_ERR, [yAfter, y], 3, R2 + 1), 4, R2 + 1, { expect: 0 });
  const triesAfter = ws.variable('tries', { value: 0 }, 0, R2 + 1.5);
  ws.display(triesAfter, 1, R2 + 1.5, { expect: search.tries });
  return { ws, viewport: { x: 0, y: 0, zoom: 0.45 }, search };
}

// ---------- Part 2: 기울기와 역전파 ----------
//
// 페이지 계산 규칙을 흉내내는 작은 시뮬레이터: 배열 원소는 float32 로 저장되고, 숫자와
// 합·내적·행렬×벡터의 누적은 double 입니다 (js/values.js).
const f32 = Math.fround;
const isVec = v => Array.isArray(v);
function zip(a, b, op) {
  if (isVec(a) && isVec(b)) return a.map((x, i) => f32(op(x, b[i])));
  if (isVec(a)) return a.map(x => f32(op(x, b)));
  if (isVec(b)) return b.map(y => f32(op(a, y)));
  return op(a, b);
}
const sim = {
  add: (a, b) => zip(a, b, (x, y) => x + y),
  sub: (a, b) => zip(a, b, (x, y) => x - y),
  mul: (a, b) => zip(a, b, (x, y) => x * y),
  div: (a, b) => zip(a, b, (x, y) => x / y),
  max: (a, b) => zip(a, b, Math.max),
  exp: a => isVec(a) ? a.map(x => f32(Math.exp(x))) : Math.exp(a),
  sum: a => a.reduce((s, x) => s + x, 0),
  dot: (a, b) => a.reduce((s, x, i) => s + x * b[i], 0),
  // m: { rows, cols, data } 행 우선
  matvec: (m, v) => Array.from({ length: m.rows }, (_, r) => { let s = 0; for (let c = 0; c < m.cols; c++) s += m.data[r * m.cols + c] * v[c]; return f32(s); }),
  tmatvec: (m, v) => Array.from({ length: m.cols }, (_, c) => { let s = 0; for (let r = 0; r < m.rows; r++) s += m.data[r * m.cols + c] * v[r]; return f32(s); }),
  outer: (a, b) => ({ rows: a.length, cols: b.length, data: a.flatMap(x => b.map(y => f32(x * y))) })
};
// 블록과 같은 순서로 계산합니다.
const simSigmoid = z => sim.div(1, sim.add(sim.exp(sim.sub(0, z)), 1));
const simSigmoidBack = (g, y) => sim.mul(g, sim.mul(y, sim.sub(1, y)));
const simLinear = (x1, x2, w1, w2, b) => sim.add(sim.add(sim.mul(w1, x1), sim.mul(w2, x2)), b);
const simSqErr = (yhat, y) => { const e = sim.sub(yhat, y); return sim.sum(sim.mul(e, e)); };
const simSqErrBack = (g, yhat, y) => sim.mul(g, sim.mul(2, sim.sub(yhat, y)));
const simStep = z => sim.sub(1, zip(sim.max(z, 0), 0, (a, b) => (a === b ? 1 : 0)));
const round3 = v => Math.round(v * 1000) / 1000;

// σ(x) = 1 / (1 + e^(−x)).  역전파: ∂x = g · y(1 − y)  (y = σ(x), 순전파 출력)
const SIGMOID = defineBlock({
  id: 'sigmoid', name: '시그모이드 · σ(x) = 1 / (1 + e⁻ˣ)', formula: '1 / (1 + e⁻ˣ)',
  build(b) {
    const zero = b.add('number', { value: 0 }, 0, 1), one = b.add('number', { value: 1 }, 2, 1);
    const neg = b.add('subtract', {}, 0, 0); b.link(zero, neg, 0);
    const e = b.add('exp', {}, 1, 0); b.link(neg, e, 0);
    const den = b.add('add', {}, 2, 0); b.link(e, den, 0); b.link(one, den, 1);
    const out = b.add('divide', {}, 3, 0); b.link(one, out, 0); b.link(den, out, 1);
    b.input(neg, 1, 'x'); b.output(out);
  },
  backward(bb) {
    const one = bb.add('number', { value: 1 }, 0, 1);
    const slope = bb.op('multiply', [bb.y, bb.op('subtract', [one, bb.y], 0, 0)], 1, 0);
    bb.outputs([bb.op('multiply', [bb.g, slope], 2, 0)]);
  }
});

// 층 하나. 역전파: ∂W = g ⊗ x, ∂x = Wᵀg, ∂b = g
const DENSE = defineBlock({
  id: 'dense', name: '층 · Wx + b', formula: 'Wx + b',
  build(b) {
    const m = b.add('matvec', {}, 0, 0);
    const out = b.add('add', {}, 1, 0); b.link(m, out, 0);
    b.input(m, 0, 'W'); b.input(m, 1, 'x'); b.input(out, 1, 'b'); b.output(out);
  },
  backward(bb) {
    const dW = bb.op('outerProduct', [bb.g, bb.input(1)], 0, 0);
    const dx = bb.op('transposeMatvec', [bb.input(0), bb.g], 0, 1);
    bb.outputs([dW, dx, bb.g]);
  }
});

// 함수 그래프를 그림(행렬)으로: 칸 (r, c) = 1  ⇔  |y_c − 높이_r| ≤ 두께.
// 같은 입력을 여러 곳에 쓰므로 안에서 값 보기로 갈라 씁니다.
const PLOT = defineBlock({
  id: 'plot', name: '그래프 그리기 · |y − 높이| ≤ 두께', formula: '[ |y_c − h_r| ≤ δ ]',
  build(b) {
    const py = b.add('display', {}, 0, 0), ph = b.add('display', {}, 0, 1);
    const zero = b.add('number', { value: 0 }, 0, 2), one = b.add('number', { value: 1 }, 0, 3);
    // 1로 채운 벡터: y·0 + 1, 높이·0 + 1
    const ones = (source, row) => {
      const zeroed = b.add('multiply', {}, 1, row); b.link(source, zeroed, 0); b.link(zero, zeroed, 1);
      const filled = b.add('add', {}, 2, row); b.link(zeroed, filled, 0); b.link(one, filled, 1);
      return filled;
    };
    const onesY = ones(py, 0), onesH = ones(ph, 1);
    const heights = b.add('outerProduct', {}, 3, 0); b.link(ph, heights, 0); b.link(onesY, heights, 1);
    const curve = b.add('outerProduct', {}, 3, 1); b.link(onesH, curve, 0); b.link(py, curve, 1);
    const d = b.add('subtract', {}, 4, 0); b.link(heights, d, 0); b.link(curve, d, 1);
    const nd = b.add('subtract', {}, 4, 1); b.link(zero, nd, 0); b.link(d, nd, 1);
    const abs = b.add('maximum', {}, 5, 0); b.link(d, abs, 0); b.link(nd, abs, 1);
    const over = b.add('subtract', {}, 6, 0); b.link(abs, over, 0);
    const clip = b.add('maximum', {}, 7, 0); b.link(over, clip, 0); b.link(zero, clip, 1);
    const hit = b.add('equal', {}, 8, 0); b.link(clip, hit, 0); b.link(zero, hit, 1);
    b.input(py, 0, 'y'); b.input(ph, 0, '높이 목록'); b.input(over, 1, '두께'); b.output(hit);
  }
});

// 기본 블록 '곱하기'의 역전파 정의: ∂a = g·b, ∂b = g·a
const MULTIPLY_BACKWARD = {
  type: 'multiply',
  definition: defineBackward(2, bb => {
    bb.outputs([bb.op('multiply', [bb.g, bb.input(1)], 0, 0), bb.op('multiply', [bb.g, bb.input(0)], 0, 1)]);
  })
};

function backwardParams2(upstream, gradients) {
  const params = { manualBackpropMode: 'backward', manualBackpropUpstream: upstream };
  gradients.forEach((name, k) => { params[`manualBackpropGradient${k}`] = name; });
  return params;
}

// ---------- 5단계: 경사하강 ----------

const GD = { target: 3, start: 0, steps: 50, good: 0.1, bad: 1.1 };

function simulateGd(lr) {
  let w = GD.start;
  for (let i = 0; i < GD.steps; i++) {
    const g = 2 * (w - GD.target);
    w = w - lr * g;
  }
  return { w, loss: (w - GD.target) * (w - GD.target) };
}

function build05() {
  const ws = new Workspace();
  const three = ws.number(GD.target, 0, 0);
  const two = ws.number(2, 0, 1);
  const rows = [];
  [['w', GD.good, 0.5], ['u', GD.bad, 3.5]].forEach(([name, lr, R]) => {
    const v = ws.variable(name, { value: GD.start, trainable: 'yes' }, 1, R);
    const eta = ws.number(lr, 1, R + 1);
    const d = ws.op('subtract', [v, three], 2, R);
    ws.display(ws.op('multiply', [d, d], 3, R), 4, R, { expect: (GD.start - GD.target) ** 2 });
    const g = ws.op('multiply', [two, d], 3, R + 1);
    const set = ws.op('setVariable', [ws.custom(SGD_STEP, [v, eta, g], 4, R + 1)], 5, R + 1, { variable: name });
    rows.push({ name, lr, set, R });
  });
  const body = ws.op('sequence', [rows[0].set, rows[1].set], 6, 1);
  ws.op('repeat', [body], 7, 1, { count: GD.steps, start: 0, indexVariable: 'i' });
  // 반복 뒤에 만든 변수 블록이라 학습 후 값을 읽습니다.
  for (const row of rows) {
    const result = simulateGd(row.lr);
    const after = ws.variable(row.name, { value: GD.start, trainable: 'yes' }, 8, row.R);
    ws.display(after, 9, row.R, { approx: result.w, tol: Math.max(1e-9, Math.abs(result.w) * 1e-9) });
    const d = ws.op('subtract', [after, three], 9, row.R + 1);
    ws.display(ws.op('multiply', [d, d], 10, row.R + 1), 11, row.R + 1, { approx: result.loss, tol: Math.max(1e-9, result.loss * 1e-9) });
  }
  return { ws, viewport: { x: 0, y: 0, zoom: 0.55 }, good: simulateGd(GD.good), bad: simulateGd(GD.bad) };
}

// ---------- 6단계: 매끄러운 활성화 함수 ----------

const PLOT_X = Array.from({ length: 61 }, (_, i) => round3(-6 + 0.2 * i));
const levels = (top, count = 41) => Array.from({ length: count }, (_, i) => round3(top - (top / (count - 1)) * i));
const DIFF_H = 0.01;

function build06() {
  const ws = new Workspace();
  const xs = ws.variable('x', { mode: 'vector', length: PLOT_X.length, value: 0 }, 0, 0, PLOT_X);
  const h01 = ws.variable('높이0_1', { mode: 'vector', length: 41, value: 0 }, 0, 1, levels(1));
  const hDer = ws.variable('높이0_0.3', { mode: 'vector', length: 41, value: 0 }, 0, 2, levels(0.3));
  const hRelu = ws.variable('높이0_6', { mode: 'vector', length: 41, value: 0 }, 0, 3, levels(6));
  const w01 = ws.number(0.03, 0, 4), wDer = ws.number(0.009, 0, 5), wRelu = ws.number(0.18, 0, 6);
  const h = ws.number(DIFF_H, 0, 7), twoH = ws.number(2 * DIFF_H, 0, 8);
  const image = { view: 'image' };
  const plot = (y, heights, width, col, row) => ws.display(ws.custom(PLOT, [y, heights, width], col, row), col + 1, row, null, image);
  // 수치 미분 (f(x + h) − f(x − h)) / 2h
  const xPlus = ws.op('add', [xs, h], 1, 7.5), xMinus = ws.op('subtract', [xs, h], 1, 8.5);
  const numeric = (block, col, row) => ws.op('divide', [ws.op('subtract', [ws.custom(block, [xPlus], col, row), ws.custom(block, [xMinus], col, row + 1)], col + 1, row), twoH], col + 2, row);

  const x = Float32Array.from(PLOT_X);
  const sigma = simSigmoid(Array.from(x));
  const slope = sim.mul(sigma, sim.sub(1, sigma));

  // 위: 계단 함수와 그 기울기
  plot(ws.custom(STEP, [xs], 2, 0), h01, w01, 3, 0);
  plot(numeric(STEP, 2, 1.5), h01, w01, 5, 1.5);
  // 가운데: 시그모이드, 기울기 σ(1 − σ), 수치 미분과 비교
  const s = ws.custom(SIGMOID, [xs], 2, 4);
  ws.display(s, 3, 3.2, { approx: sigma, tol: 1e-6 });
  plot(s, h01, w01, 3, 4);
  const one = ws.number(1, 2, 5.2);
  const ds = ws.op('multiply', [s, ws.op('subtract', [one, s], 3, 5.2)], 4, 5.2);
  ws.display(ds, 5, 5.2, { approx: slope, tol: 1e-6 });
  plot(ds, hDer, wDer, 6, 5.2);
  const gap = ws.op('subtract', [numeric(SIGMOID, 2, 7.5), ds], 5, 7.5);
  const absGap = ws.op('maximum', [gap, ws.op('subtract', [ws.number(0, 5, 8.5), gap], 6, 8.5)], 6, 7.5);
  ws.display(ws.op('arrayMax', [absGap], 7, 7.5), 8, 7.5, { approx: 0, tol: 1e-4 });
  // 아래: ReLU 와 그 기울기(= 계단 함수)
  const r = ws.custom(RELU, [xs], 2, 10);
  plot(r, hRelu, wRelu, 3, 10);
  plot(ws.custom(STEP, [xs], 5, 10), h01, w01, 6, 10);
  return { ws, viewport: { x: 0, y: 0, zoom: 0.5 } };
}

// ---------- 7단계: 연쇄 법칙과 수동 역전파 ----------

const CHAIN = { w: 0.5, x: 2, h: 0.001 };

function build07() {
  const ws = new Workspace();
  const w = ws.variable('w', { value: CHAIN.w }, 0, 0);
  const x = ws.variable('x', { value: CHAIN.x }, 0, 1);
  const z = ws.op('multiply', [w, x], 1, 0);
  const y = ws.custom(SIGMOID, [z], 2, 0);
  const sig = v => 1 / (1 + Math.exp(-v));
  const zv = CHAIN.w * CHAIN.x, yv = sig(zv), slope = yv * (1 - yv);
  ws.display(z, 2, 1, { approx: zv, tol: 1e-12 });
  ws.display(y, 3, 0, { approx: yv, tol: 1e-12 });

  // 역전파: 출력 쪽부터. g = ∂y/∂y = 1 → σ 역전파가 dz 를, 곱하기 역전파가 dw, dx 를 씁니다.
  ws.variable('g', { value: 1 }, 0, 3);
  const backSig = ws.custom(SIGMOID, [z], 2, 3, backwardParams2('g', ['dz']));
  const backMul = ws.op('multiply', [w, x], 1, 4.2, backwardParams2('dz', ['dw', 'dx']));
  ws.op('sequence', [backSig, backMul], 3, 3.6);
  // 역전파 블록보다 나중에 만든 변수 블록이라 기록된 기울기를 읽습니다.
  ws.display(ws.variable('dz', { value: 0 }, 4, 3), 5, 3, { approx: slope, tol: 1e-6 });
  ws.display(ws.variable('dw', { value: 0 }, 4, 4), 5, 4, { approx: slope * CHAIN.x, tol: 1e-6 });
  ws.display(ws.variable('dx', { value: 0 }, 4, 5), 5, 5, { approx: slope * CHAIN.w, tol: 1e-6 });

  // 확인: w 를 ±h 만큼 움직인 수치 미분
  const h = ws.number(CHAIN.h, 0, 7);
  const yAt = (wv, row) => ws.custom(SIGMOID, [ws.op('multiply', [wv, x], 2, row)], 3, row);
  const up = yAt(ws.op('add', [w, h], 1, 7), 7), down = yAt(ws.op('subtract', [w, h], 1, 8), 8);
  const numeric = ws.op('divide', [ws.op('subtract', [up, down], 4, 7.5), ws.number(2 * CHAIN.h, 4, 8.5)], 5, 7.5);
  const expected = (sig((CHAIN.w + CHAIN.h) * CHAIN.x) - sig((CHAIN.w - CHAIN.h) * CHAIN.x)) / (2 * CHAIN.h);
  ws.display(numeric, 6, 7.5, { approx: expected, tol: 1e-9 });
  return { ws, viewport: { x: 0, y: 0, zoom: 0.6 }, builtins: [MULTIPLY_BACKWARD], chain: { zv, yv, slope } };
}

// ---------- 8·9단계 공통: 한 회차 = 순전파 → 역전파 → 값 바꾸기 ----------

// 역전파 실행 블록을 순서대로 묶고, 그 뒤에 값 바꾸기들을 묶어 반복합니다.
function trainingLoop(ws, backward, updates, col, row, count) {
  const body = ws.op('sequence', [ws.chain(backward, col, row), ws.chain(updates, col, row + 1)], col + backward.length, row);
  return ws.op('repeat', [body], col + backward.length + 1, row, { count, start: 0, indexVariable: 'i' });
}

// ---------- 8단계: 뉴런 하나를 경사하강으로 (AND) ----------

const NEURON = { lr: 2, steps: 500, init: { w1: 0, w2: 0, b: 0 } };
const AND_TABLE = [0, 0, 0, 1];

function simulateNeuron() {
  let { w1, w2, b } = NEURON.init;
  const x1 = TRUTH.x1, x2 = TRUTH.x2, t = AND_TABLE;
  const forward = () => simSigmoid(simLinear(x1, x2, w1, w2, b));
  const before = forward();
  for (let i = 0; i < NEURON.steps; i++) {
    const y = forward();
    const dz = simSigmoidBack(simSqErrBack(1, y, t), y);
    const d = [sim.sum(sim.mul(dz, x1)), sim.sum(sim.mul(dz, x2)), sim.sum(dz)];
    w1 = w1 - NEURON.lr * d[0]; w2 = w2 - NEURON.lr * d[1]; b = b - NEURON.lr * d[2];
  }
  const after = forward();
  return { before, lossBefore: simSqErr(before, t), w1, w2, b, after, lossAfter: simSqErr(after, t) };
}

function build08() {
  const ws = new Workspace();
  const sim8 = simulateNeuron();
  const { x1, x2 } = truthInputs(ws, 0, 0, false);
  const t = ws.variable('t', { mode: 'vector', length: 4, value: 0 }, 0, 2, AND_TABLE);
  const params = ['w1', 'w2', 'b'].map((name, k) => ws.variable(name, { value: NEURON.init[name], trainable: 'yes' }, 0, 3 + k));
  const eta = ws.number(NEURON.lr, 0, 6);

  // 순전파
  const z = ws.custom(LINEAR, [x1, x2, ...params], 1.2, 1);
  const y = ws.custom(SIGMOID, [z], 2.2, 1);
  ws.display(y, 3.2, 0, { approx: sim8.before, tol: 1e-6 });
  const loss = ws.custom(SQ_ERR, [y, t], 3.2, 1);
  ws.display(loss, 4.2, 1, { approx: sim8.lossBefore, tol: 1e-6 });

  // 역전파 (손실 → σ → 선형 모형)와 갱신
  const gradNames = ['dw1', 'dw2', 'db'];
  ['gL', 'dy', 'dz', ...gradNames].forEach((name, k) => ws.variable(name, { value: name === 'gL' ? 1 : 0 }, 5.4 + (k % 3), 3 + Math.floor(k / 3) * 0.8));
  const backward = [
    ws.custom(SQ_ERR, [y, t], 3.2, 4.8, backwardParams2('gL', ['dy', ''])),
    ws.custom(SIGMOID, [z], 2.2, 4.8, backwardParams2('dy', ['dz'])),
    ws.custom(LINEAR, [x1, x2, ...params], 1.2, 4.8, backwardParams2('dz', ['', '', ...gradNames]))
  ];
  const updates = params.map((p, k) => {
    const grad = ws.variable(gradNames[k], { value: 0 }, 1.2 + k, 7);
    return ws.op('setVariable', [ws.custom(SGD_STEP, [p, eta, grad], 1.2 + k, 8)], 1.2 + k, 9, { variable: ['w1', 'w2', 'b'][k] });
  });
  trainingLoop(ws, backward, updates, 4.6, 6.4, NEURON.steps);

  // 학습 후
  const A = 11;
  const after = ['w1', 'w2', 'b'].map((name, k) => ws.variable(name, { value: NEURON.init[name], trainable: 'yes' }, 0, A + k));
  after.forEach((v, k) => ws.display(v, 1, A + k, { approx: [sim8.w1, sim8.w2, sim8.b][k], tol: 1e-4 }));
  const yAfter = ws.custom(SIGMOID, [ws.custom(LINEAR, [x1, x2, ...after], 2, A)], 3, A);
  ws.display(yAfter, 4, A, { approx: sim8.after, tol: 1e-4 });
  ws.display(ws.custom(SQ_ERR, [yAfter, t], 4, A + 1), 5, A + 1, { approx: sim8.lossAfter, tol: 1e-4 });
  return { ws, viewport: { x: 0, y: 0, zoom: 0.5 }, sim: sim8 };
}

// ---------- 9단계: 2-2-1 신경망으로 XOR 학습 ----------

const XOR_NET = { lr: 2, steps: 3000, scale: 1 };
const XOR_PARAMS = ['w11', 'w12', 'b1', 'w21', 'w22', 'b2', 'v1', 'v2', 'c'];

function xorForward(p) {
  const h1 = simSigmoid(simLinear(TRUTH.x1, TRUTH.x2, p.w11, p.w12, p.b1));
  const h2 = simSigmoid(simLinear(TRUTH.x1, TRUTH.x2, p.w21, p.w22, p.b2));
  return { h1, h2, y: simSigmoid(simLinear(h1, h2, p.v1, p.v2, p.c)) };
}

function simulateXorNet(init) {
  const p = { ...init };
  const t = TRUTH.xor;
  const before = xorForward(p).y;
  for (let i = 0; i < XOR_NET.steps; i++) {
    const { h1, h2, y } = xorForward(p);
    const dzo = simSigmoidBack(simSqErrBack(1, y, t), y);
    const dz1 = simSigmoidBack(sim.mul(dzo, p.v1), h1);
    const dz2 = simSigmoidBack(sim.mul(dzo, p.v2), h2);
    const g = {
      w11: sim.sum(sim.mul(dz1, TRUTH.x1)), w12: sim.sum(sim.mul(dz1, TRUTH.x2)), b1: sim.sum(dz1),
      w21: sim.sum(sim.mul(dz2, TRUTH.x1)), w22: sim.sum(sim.mul(dz2, TRUTH.x2)), b2: sim.sum(dz2),
      v1: sim.sum(sim.mul(dzo, h1)), v2: sim.sum(sim.mul(dzo, h2)), c: sim.sum(dzo)
    };
    for (const name of XOR_PARAMS) p[name] = p[name] - XOR_NET.lr * g[name];
  }
  const after = xorForward(p).y;
  return { init, before, lossBefore: simSqErr(before, t), params: p, after, lossAfter: simSqErr(after, t) };
}

// 초깃값: 고정 시드 난수를 소수 셋째 자리로 반올림. 학습이 성공하는 첫 시드를 고릅니다.
function chooseXorInit() {
  for (let seed = 1; seed < 200; seed++) {
    const r = lcgFloat32(XOR_PARAMS.length, seed, XOR_NET.scale);
    const init = Object.fromEntries(XOR_PARAMS.map((name, k) => [name, round3(r[k])]));
    const result = simulateXorNet(init);
    if (result.lossAfter < 0.02) return { seed, ...result };
  }
  throw new Error('XOR 을 학습하는 초깃값을 찾지 못했습니다.');
}

function build09() {
  const ws = new Workspace();
  const sim9 = chooseXorInit();
  const { x1, x2 } = truthInputs(ws, 0, 0, false);
  const t = ws.variable('t', { mode: 'vector', length: 4, value: 0 }, 0, 2, TRUTH.xor);
  const P = Object.fromEntries(XOR_PARAMS.map((name, k) => [name, ws.variable(name, { value: sim9.init[name], trainable: 'yes' }, 0, 3 + k * 0.75)]));
  const eta = ws.number(XOR_NET.lr, 0, 10);

  // 순전파: 은닉 뉴런 h1, h2 → 출력 뉴런 y
  const z1 = ws.custom(LINEAR, [x1, x2, P.w11, P.w12, P.b1], 1.3, 0);
  const h1 = ws.custom(SIGMOID, [z1], 2.3, 0);
  const z2 = ws.custom(LINEAR, [x1, x2, P.w21, P.w22, P.b2], 1.3, 1.6);
  const h2 = ws.custom(SIGMOID, [z2], 2.3, 1.6);
  const zo = ws.custom(LINEAR, [h1, h2, P.v1, P.v2, P.c], 3.3, 0.8);
  const y = ws.custom(SIGMOID, [zo], 4.3, 0.8);
  ws.display(y, 5.3, 0, { approx: sim9.before, tol: 1e-6 });
  ws.display(ws.custom(SQ_ERR, [y, t], 5.3, 1), 6.3, 1, { approx: sim9.lossBefore, tol: 1e-6 });

  // 역전파: 손실 → 출력 σ → 출력 선형 → (은닉 σ → 은닉 선형) × 2
  const gradOf = name => `d${name}`;
  ['gL', 'dy', 'dzo', 'dh1', 'dh2', 'dz1', 'dz2'].forEach((name, k) => ws.variable(name, { value: name === 'gL' ? 1 : 0 }, 7.5 + (k % 4), 0 + Math.floor(k / 4) * 0.8));
  const B = 3.6;
  const backward = [
    ws.custom(SQ_ERR, [y, t], 5.3, B, backwardParams2('gL', ['dy', ''])),
    ws.custom(SIGMOID, [zo], 4.3, B, backwardParams2('dy', ['dzo'])),
    ws.custom(LINEAR, [h1, h2, P.v1, P.v2, P.c], 3.3, B, backwardParams2('dzo', ['dh1', 'dh2', 'dv1', 'dv2', 'dc'])),
    ws.custom(SIGMOID, [z1], 2.3, B, backwardParams2('dh1', ['dz1'])),
    ws.custom(LINEAR, [x1, x2, P.w11, P.w12, P.b1], 1.3, B, backwardParams2('dz1', ['', '', 'dw11', 'dw12', 'db1'])),
    ws.custom(SIGMOID, [z2], 2.3, B + 1.6, backwardParams2('dh2', ['dz2'])),
    ws.custom(LINEAR, [x1, x2, P.w21, P.w22, P.b2], 1.3, B + 1.6, backwardParams2('dz2', ['', '', 'dw21', 'dw22', 'db2']))
  ];
  const updates = XOR_PARAMS.map((name, k) => {
    const grad = ws.variable(gradOf(name), { value: 0 }, 1 + k, 7.6);
    return ws.op('setVariable', [ws.custom(SGD_STEP, [P[name], eta, grad], 1 + k, 8.6)], 1 + k, 9.6, { variable: name });
  });
  trainingLoop(ws, backward, updates, 6.5, 4.4, XOR_NET.steps);

  // 학습 후
  const A = 12;
  const after = Object.fromEntries(XOR_PARAMS.map((name, k) => [name, ws.variable(name, { value: sim9.init[name], trainable: 'yes' }, 0, A + k * 0.75)]));
  const a1 = ws.custom(SIGMOID, [ws.custom(LINEAR, [x1, x2, after.w11, after.w12, after.b1], 1.3, A)], 2.3, A);
  const a2 = ws.custom(SIGMOID, [ws.custom(LINEAR, [x1, x2, after.w21, after.w22, after.b2], 1.3, A + 1.6)], 2.3, A + 1.6);
  const yAfter = ws.custom(SIGMOID, [ws.custom(LINEAR, [a1, a2, after.v1, after.v2, after.c], 3.3, A + 0.8)], 4.3, A + 0.8);
  ws.display(yAfter, 5.3, A, { approx: sim9.after, tol: 1e-3 });
  ws.display(ws.custom(SQ_ERR, [yAfter, t], 5.3, A + 1), 6.3, A + 1, { approx: sim9.lossAfter, tol: 1e-3 });
  return { ws, viewport: { x: 0, y: 0, zoom: 0.45 }, sim: sim9 };
}

// ---------- 10단계: 층 블록 Wx + b 로 다시 조립 ----------

// 은닉층 W1 (2×2), b1 (2), 출력층 W2 (1×2), b2 (1). 그림 한 장(진리표 한 줄)씩 학습합니다.
const LAYER_NET = { lr: 2, epochs: 1000, scale: 1 };
const XOR_X = { rows: 2, cols: 4, data: [...TRUTH.x1, ...TRUTH.x2] };

function layerForward(p, x) {
  const z1 = sim.add(sim.matvec(p.W1, x), p.b1);
  const h = simSigmoid(z1);
  const z2 = sim.add(sim.matvec(p.W2, h), p.b2);
  return { h, y: simSigmoid(z2) };
}

function simulateLayerNet(seed) {
  const p = {
    W1: { rows: 2, cols: 2, data: Array.from(lcgFloat32(4, seed, LAYER_NET.scale)) },
    b1: [0, 0],
    W2: { rows: 1, cols: 2, data: Array.from(lcgFloat32(2, seed + 1, LAYER_NET.scale)) },
    b2: [0]
  };
  const columns = [0, 1, 2, 3].map(k => [TRUTH.x1[k], TRUTH.x2[k]]);
  const outputs = () => columns.map(x => layerForward(p, x).y[0]);
  const before = outputs();
  for (let e = 0; e < LAYER_NET.epochs; e++) {
    for (let k = 0; k < 4; k++) {
      const x = columns[k], t = TRUTH.xor[k];
      const { h, y } = layerForward(p, x);
      const dz2 = simSigmoidBack(simSqErrBack(1, y, t), y);
      const dW2 = sim.outer(dz2, h), dh = sim.tmatvec(p.W2, dz2);
      const dz1 = simSigmoidBack(dh, h);
      const dW1 = sim.outer(dz1, x);
      const step = (m, d) => ({ ...m, data: sim.sub(m.data, sim.mul(LAYER_NET.lr, d.data)) });
      Object.assign(p, { W1: step(p.W1, dW1), b1: sim.sub(p.b1, sim.mul(LAYER_NET.lr, dz1)), W2: step(p.W2, dW2), b2: sim.sub(p.b2, sim.mul(LAYER_NET.lr, dz2)) });
    }
  }
  const after = outputs();
  const loss = after.reduce((s, y, k) => s + (y - TRUTH.xor[k]) ** 2, 0);
  return { seed, before, after, loss, params: p };
}

function chooseLayerSeed() {
  for (let seed = 1; seed < 200; seed++) {
    const result = simulateLayerNet(seed);
    if (result.loss < 0.02) return result;
  }
  throw new Error('XOR 을 학습하는 층 초깃값을 찾지 못했습니다.');
}

function build10() {
  const ws = new Workspace();
  const sim10 = chooseLayerSeed();
  const X = ws.variable('X', { mode: 'matrix', rows: 2, cols: 4, value: 0 }, 0, 0, XOR_X.data);
  const T = ws.variable('t', { mode: 'vector', length: 4, value: 0 }, 0, 1, TRUTH.xor);
  const idx = ws.variable('번호', { mode: 'vector', length: 4, value: 0 }, 0, 2, [0, 1, 2, 3]);
  const k = ws.variable('k', { value: 0 }, 0, 3);
  const weight = (name, rows, cols, seed, col, row) => ws.variable(name, { mode: 'matrix', rows, cols, init: 'random', seed, scale: LAYER_NET.scale, trainable: 'yes' }, col, row);
  const bias = (name, length, col, row) => ws.variable(name, { mode: 'vector', length, value: 0, trainable: 'yes' }, col, row);
  const P = { W1: weight('W1', 2, 2, sim10.seed, 0, 4.5), b1: bias('b1', 2, 0, 5.5), W2: weight('W2', 1, 2, sim10.seed + 1, 0, 6.5), b2: bias('b2', 1, 0, 7.5) };
  const eta = ws.number(LAYER_NET.lr, 0, 8.5);

  // k번째 줄: x = X·[번호 = k],  정답 = t·[번호 = k]
  const pick = ws.op('equal', [idx, k], 1, 2.5);
  const x = ws.op('matvec', [X, pick], 2, 1);
  const target = ws.op('dot', [T, pick], 2, 2.5);
  const z1 = ws.custom(DENSE, [P.W1, x, P.b1], 3, 1);
  const h = ws.custom(SIGMOID, [z1], 4, 1);
  const z2 = ws.custom(DENSE, [P.W2, h, P.b2], 5, 1);
  const y = ws.custom(SIGMOID, [z2], 6, 1);
  ws.custom(SQ_ERR, [y, target], 7, 1);

  ['gL', 'dy', 'dz2', 'dh', 'dz1', 'dW2', 'db2', 'dW1', 'db1'].forEach((name, n) => {
    const shape = { dy: [1], dz2: [1], db2: [1], dh: [2], dz1: [2], db1: [2], dW2: [1, 2], dW1: [2, 2] }[name];
    const params = !shape ? { value: 1 } : shape.length === 1 ? { mode: 'vector', length: shape[0], value: 0 } : { mode: 'matrix', rows: shape[0], cols: shape[1], value: 0 };
    ws.variable(name, params, 8.5 + (n % 3), Math.floor(n / 3) * 0.8);
  });
  const B = 3.6;
  const backward = [
    ws.custom(SQ_ERR, [y, target], 7, B, backwardParams2('gL', ['dy', ''])),
    ws.custom(SIGMOID, [z2], 6, B, backwardParams2('dy', ['dz2'])),
    ws.custom(DENSE, [P.W2, h, P.b2], 5, B, backwardParams2('dz2', ['dW2', 'dh', 'db2'])),
    ws.custom(SIGMOID, [z1], 4, B, backwardParams2('dh', ['dz1'])),
    ws.custom(DENSE, [P.W1, x, P.b1], 3, B, backwardParams2('dz1', ['dW1', '', 'db1']))
  ];
  const updates = ['W1', 'b1', 'W2', 'b2'].map((name, n) => {
    const gradParams = { W1: { mode: 'matrix', rows: 2, cols: 2, value: 0 }, b1: { mode: 'vector', length: 2, value: 0 }, W2: { mode: 'matrix', rows: 1, cols: 2, value: 0 }, b2: { mode: 'vector', length: 1, value: 0 } }[name];
    const grad = ws.variable(`d${name}`, gradParams, 2 + n, 6.2);
    return ws.op('setVariable', [ws.custom(SGD_STEP, [P[name], eta, grad], 2 + n, 7.2)], 2 + n, 8.2, { variable: name });
  });
  const body = ws.op('sequence', [ws.chain(backward, 6.5, 6.2), ws.chain(updates, 6.5, 7.2)], 11, 6.2);
  const rows = ws.op('repeat', [body], 12, 6.2, { count: 4, start: 0, indexVariable: 'k' });
  ws.op('repeat', [rows], 13, 6.2, { count: LAYER_NET.epochs, start: 0, indexVariable: 'epoch' });
  ws.variable('epoch', { value: 0 }, 13, 7.2);

  // 학습 후: 진리표 네 줄을 하나씩 넣어 봅니다.
  const A = 10.5;
  const after = {
    W1: weight('W1', 2, 2, sim10.seed, 0, A), b1: bias('b1', 2, 0, A + 1),
    W2: weight('W2', 1, 2, sim10.seed + 1, 0, A + 2), b2: bias('b2', 1, 0, A + 3)
  };
  [0, 1, 2, 3].forEach(n => {
    const xn = ws.variable(`x${n + 1}`, { mode: 'vector', length: 2, value: 0 }, 1, A + n, [TRUTH.x1[n], TRUTH.x2[n]]);
    const out = ws.custom(SIGMOID, [ws.custom(DENSE, [after.W2, ws.custom(SIGMOID, [ws.custom(DENSE, [after.W1, xn, after.b1], 2, A + n)], 3, A + n), after.b2], 4, A + n)], 5, A + n);
    ws.display(out, 6, A + n, { approx: [sim10.after[n]], tol: 1e-3 });
  });
  return { ws, viewport: { x: 0, y: 0, zoom: 0.45 }, sim: sim10 };
}

// ---------- CNN: Quick Draw 15종 분류 ----------
//
// source/quickdraw-cnn.mmlab(편집 모드에서 만든 학습 작업공간)과 source/quickdraw-cnn.mmlweights.json
// (아래처럼 정리한 cnn-train 단계를 epoch 반복만 60으로 바꿔 학습한 가중치)을 읽어 발표용으로 다시
// 배치합니다. 사용자 블록 정의, 변수 설정,
// 순전파·역전파·모멘텀 갱신의 계산은 원본 그대로이고 바뀌는 것은 다음뿐입니다.
//   · 복제된 평가용 순전파와 혼동행렬 평가 반복을 빼고, 학습 순전파의 점수에 Softmax를 달아
//     '예측 확률' 출력으로 씁니다(그림별 예측 평가 · 손그림 분류 패널에서 선택).
//   · 둘 다 계산 사슬 103개를 단계별 '차례로 실행' 사용자 블록 5개로 묶습니다.
//   · 가중치 갱신이 이번 배치의 새 모멘텀을 쓰도록 연결합니다. 원본은 값 바꾸기가 회차 끝에 반영되는
//     규칙 때문에 한 배치 전의 모멘텀으로 갱신했습니다.
//   · 파라미터 21개를 한 줄씩 [누적 · 초기화 · 모멘텀 · 갱신] 표로 배치합니다.

function readSource(name) { return JSON.parse(fs.readFileSync(path.join(HERE, 'source', name), 'utf8')); }
const CNN_SOURCE = readSource('quickdraw-cnn.mmlab');
const CNN_WEIGHTS = readSource('quickdraw-cnn.mmlweights.json');
const CNN_BRANCHES = 8;
const CNN_PARAMS = [
  ...Array.from({ length: CNN_BRANCHES }, (_, i) => `K${i + 1}`),
  ...Array.from({ length: CNN_BRANCHES }, (_, i) => `W1_${i + 1}`),
  'b1', 'W2', 'b2', 'W3', 'b3'
];
const CNN_PARAM_COUNT = CNN_WEIGHTS.variables.reduce((sum, v) => sum + v.value.shape.reduce((a, b) => a * b, 1), 0);

// 그림별 예측 평가로 페이지에서 측정한 정확도 (테스트 구간 15,000~29,999).
// 생성기에서 흉내내기에는 계산이 커서 측정값을 적어 두고, 내용 검사가 다시 측정해 비교합니다.
// 1 epoch 학습 후 정확도는 무작위 초기값에서 이 단계의 '계산'을 한 번 실행한 결과입니다.
const CNN_EVAL = { start: 15000, count: 15000 };
const CNN_MEASURED = { initial: 0.06686666666666667, oneEpoch: 0.8132666666666667, trained: 0.8787333333333334 };
const percent = value => `${(value * 100).toFixed(1)}%`;

function sourceVariableParams(name) {
  const node = CNN_SOURCE.graph.nodes.find(n => n.type === 'variable' && n.params.name === name);
  if (!node) throw new Error(`원본에 '${name}' 변수가 없습니다.`);
  return JSON.parse(JSON.stringify(node.params));
}

function sourceBlock(id) {
  const definition = CNN_SOURCE.userBlocks.find(block => block.id === id);
  if (!definition) throw new Error(`원본에 '${id}' 사용자 블록이 없습니다.`);
  return { id, uses: [], definition };
}

const CNN_BLOCKS = Object.fromEntries(['conv5_pool12', 'dense8_relu128', 'dense_relu64', 'dense_linear15', 'softmax_ce15', 'softmax15', 'lr_decay60', 'momentum15', 'weight_step']
  .map(id => [id, sourceBlock(id)]));

// 둘 다 계산 사슬을 묶은 블록. 입력을 위에서부터 차례로 계산하고 마지막 결과를 내보냅니다.
function runInOrderBlock(id, name, labels) {
  return defineBlock({
    id, name, formula: `${labels.length}개를 위에서부터 차례로 계산`,
    build(b) {
      let last = b.add('sequence', {}, 0, 0);
      b.input(last, 0, labels[0]); b.input(last, 1, labels[1]);
      for (let k = 2; k < labels.length; k++) {
        const next = b.add('sequence', {}, (k - 1) * 0.7, (k - 1) * 0.6);
        b.link(last, next, 0); b.input(next, 1, labels[k]);
        last = next;
      }
      b.output(last);
    }
  });
}

const RUN_BACKWARD = runInOrderBlock('run_backward', '역전파 · 출력층부터 차례로', [
  '손실', '출력층 15', '은닉층 64', '은닉층 128', ...Array.from({ length: CNN_BRANCHES }, (_, i) => `합성곱 ${i + 1}`)
]);
const RUN_ACCUMULATE = runInOrderBlock('run_accumulate', '기울기 누적 · g ← g + ∂L/∂p', CNN_PARAMS.map(p => `g${p}`));
const RUN_UPDATE = runInOrderBlock('run_update', '파라미터 갱신 · 모멘텀 → 가중치', CNN_PARAMS.flatMap(p => [`v_${p}`, p]));
const RUN_RESET = runInOrderBlock('run_reset', '기울기 초기화 · g ← 0', CNN_PARAMS.map(p => `g${p}`));

function zeroLike(ws, params, x, y) {
  return params.mode === 'matrix'
    ? ws.add('matrix', { rows: params.rows, cols: params.cols, init: 'constant', value: 0, seed: 1, scale: 0.05 }, x / COL, y / ROW)
    : ws.add('constantVector', { length: params.length, value: 0 }, x / COL, y / ROW);
}

function backwardParams(upstream, gradients) {
  const params = { manualBackpropMode: 'backward', manualBackpropUpstream: upstream };
  gradients.forEach((name, k) => { params[`manualBackpropGradient${k}`] = name; });
  return params;
}

// trained = true 이면 학습된 가중치를 저장된 런타임 값으로 함께 넣습니다.
function buildCnn({ trained }) {
  const ws = new Workspace();
  const at = (x, y) => [x / COL, y / ROW];
  const variable = (name, x, y) => ws.add('variable', sourceVariableParams(name), ...at(x, y));
  const op = (type, inputs, x, y, params = {}) => ws.op(type, inputs, ...at(x, y), params);
  const custom = (id, inputs, x, y, params = {}) => {
    ws.use(CNN_BLOCKS[id] ?? id);
    const nodeId = ws.add(`custom:${(CNN_BLOCKS[id] ?? id).id}`, params, ...at(x, y));
    inputs.forEach((from, k) => ws.link(from, nodeId, k));
    return nodeId;
  };

  // ---- 데이터: 학습 그림 번호 = batch × 15 + k ----
  const batch = variable('batch', 0, 0);
  const fifteen = ws.number(15, ...at(0, 190));
  const k = variable('k', 0, 380);
  const dataset = ws.add('dataset', {}, ...at(0, 570));
  const index = op('add', [op('multiply', [batch, fifteen], 300, 90), k], 600, 300);
  const sample = op('datasetSampleByIndex', [dataset, index], 900, 560);
  const image = op('sampleImage', [sample], 1200, 760);
  const oneHot = op('sampleOneHot', [sample], 1200, 1900);

  // ---- 순전파: 합성곱 8분기 → Dense 128 → Dense 64 → Dense 15 ----
  const P = Object.fromEntries(CNN_PARAMS.map(name => [name, null]));
  const conv = [];
  for (let i = 0; i < CNN_BRANCHES; i++) {
    P[`K${i + 1}`] = variable(`K${i + 1}`, 1500, i * 240 + 60);
    conv.push(custom('conv5_pool12', [image, P[`K${i + 1}`]], 1800, i * 240));
  }
  for (let i = 0; i < CNN_BRANCHES; i++) P[`W1_${i + 1}`] = variable(`W1_${i + 1}`, 2150, i * 240);
  P.b1 = variable('b1', 2150, 1920);
  const h1 = custom('dense8_relu128', [...conv.map((_, i) => P[`W1_${i + 1}`]), ...conv, P.b1], 2500, 300);
  P.W2 = variable('W2', 2850, 120); P.b2 = variable('b2', 2850, 900);
  const h2 = custom('dense_relu64', [P.W2, h1, P.b2], 3150, 500);
  P.W3 = variable('W3', 3450, 120); P.b3 = variable('b3', 3450, 900);
  const logits = custom('dense_linear15', [P.W3, h2, P.b3], 3750, 500);
  const loss = custom('softmax_ce15', [logits, oneHot], 4100, 500);
  const probability = custom('softmax15', [logits], 4100, 1000);
  // '계산'은 학습 반복 전에 이 값들을 한 번 계산합니다. 학습 후 값은 '선택 계산'으로 봅니다.
  ws.display(loss, ...at(4450, 500));
  ws.display(probability, ...at(4450, 1000));

  // ---- 역전파: 순전파 바로 아래에 같은 열로 ----
  const BY = 2300;
  const gradientVariables = ['gLoss', 'dLogits', 'dh2', 'dh1', ...Array.from({ length: CNN_BRANCHES }, (_, i) => `dP${i + 1}`)];
  gradientVariables.forEach((name, i) => variable(name, 4500 + Math.floor(i / 6) * 300, BY + (i % 6) * 220));
  const backward = [
    custom('softmax_ce15', [logits, oneHot], 4100, BY, backwardParams('gLoss', ['dLogits', ''])),
    custom('dense_linear15', [P.W3, h2, P.b3], 3750, BY, backwardParams('dLogits', ['dW3_local', 'dh2', 'db3_local'])),
    custom('dense_relu64', [P.W2, h1, P.b2], 3150, BY, backwardParams('dh2', ['dW2_local', 'dh1', 'db2_local'])),
    custom('dense8_relu128', [...conv.map((_, i) => P[`W1_${i + 1}`]), ...conv, P.b1], 2500, BY, backwardParams('dh1', [
      ...conv.map((_, i) => `dW1_${i + 1}_local`), ...conv.map((_, i) => `dP${i + 1}`), 'db1_local'
    ])),
    ...conv.map((_, i) => custom('conv5_pool12', [image, P[`K${i + 1}`]], 1800, BY + i * 240, backwardParams(`dP${i + 1}`, ['', `dK${i + 1}_local`])))
  ];
  const runBackward = custom(RUN_BACKWARD, backward, 5200, BY);

  // ---- 파라미터 21개: 한 줄에 하나씩 ----
  const TY = BY + CNN_BRANCHES * 240 + 500, RH = 230;
  const epoch = variable('epoch', 5100, TY - 420);
  const lr = custom('lr_decay60', [epoch], 5400, TY - 420);
  const accumulate = [], reset = [], update = [];
  CNN_PARAMS.forEach((name, row) => {
    const y = TY + row * RH;
    const shape = sourceVariableParams(name);
    // 누적: g ← g + (이번 그림의 기울기)
    const local = variable(`d${name}_local`, 0, y);
    const g = variable(`g${name}`, 300, y);
    accumulate.push(op('setVariable', [op('add', [g, local], 600, y)], 900, y, { variable: `g${name}` }));
    // 초기화: g ← 0
    reset.push(op('setVariable', [zeroLike(ws, shape, 1500, y)], 1800, y, { variable: `g${name}` }));
    // 갱신: v ← 0.9v + g/15,  p ← p − ηv
    const v = variable(`v_${name}`, 2700, y);
    const momentum = custom('momentum15', [v, g], 3000, y);
    update.push(op('setVariable', [momentum], 3300, y, { variable: `v_${name}` }));
    const param = variable(name, 3700, y);
    update.push(op('setVariable', [custom('weight_step', [param, momentum, lr], 4000, y)], 4300, y, { variable: name }));
  });
  const midRow = TY + (CNN_PARAMS.length / 2) * RH;
  const runAccumulate = custom(RUN_ACCUMULATE, accumulate, 1150, midRow - 600);
  const runReset = custom(RUN_RESET, reset, 2100, midRow - 600);
  const runUpdate = custom(RUN_UPDATE, update, 4650, midRow - 1300);

  // ---- 학습 반복: 그림 15장마다 기울기를 모아 한 번 갱신 ----
  const perSample = op('sequence', [runBackward, runAccumulate], 5500, BY);
  const samples = op('repeat', [perSample], 5800, BY, { count: 15, start: 0, indexVariable: 'k' });
  const perBatch = op('sequence', [op('sequence', [samples, runUpdate], 6100, BY), runReset], 6400, BY);
  const batches = op('repeat', [perBatch], 6700, BY, { count: 1000, start: 0, indexVariable: 'batch' });
  op('repeat', [batches], 7000, BY, { count: 1, start: 0, indexVariable: 'epoch' });

  if (trained) {
    for (const saved of CNN_WEIGHTS.variables) {
      ws.runtimeVariables.push({ name: saved.name, signature: variableSignature(sourceVariableParams(saved.name)), value: saved.value });
    }
  }
  // 내용 검사: 예측 확률 출력으로 테스트 구간 정확도를 재고, train 이면 '계산'(1 epoch) 뒤에 다시 잽니다.
  const verify = trained
    ? { probabilityId: probability, ...CNN_EVAL, before: CNN_MEASURED.trained, tol: 0.002 }
    : { probabilityId: probability, ...CNN_EVAL, before: CNN_MEASURED.initial, tol: 0.002, train: true, after: CNN_MEASURED.oneEpoch, afterTol: 0.01 };
  return { ws, datasetSelection: CNN_SOURCE.datasetSelection, verify, viewport: { x: 20, y: 20, zoom: 0.32 } };
}

// ---------- Part 3: 실제 데이터 (Quick Draw 15종) ----------
//
// 데이터 번호 i → 종류 i mod 15 의 ⌊i / 15⌋번째 그림 (데이터 샘플 블록).
// 학습용 0~14,999 (종류마다 1,000장), 평가용 15,000~29,999.

const QD_CLASSES = CNN_SOURCE.datasetSelection;
const QD = { classes: QD_CLASSES.length, pixels: 784, train: 15000, test: { start: 15000, count: 15000 } };
let qdBytes = null;

// js/dataset.js getNormalizedImage 와 같은 값: 바이트 / 255 를 float32 로.
function qdImage(i) {
  qdBytes ??= QD_CLASSES.map(name => fs.readFileSync(path.join(HERE, '..', 'data', `${name}.bin`)));
  const label = i % QD.classes, k = Math.floor(i / QD.classes);
  const bytes = qdBytes[label], x = new Float32Array(QD.pixels);
  for (let j = 0; j < QD.pixels; j++) x[j] = bytes[k * QD.pixels + j] / 255;
  return { x, label };
}
const oneHot = (index, length = QD.classes) => Array.from({ length }, (_, k) => (k === index ? 1 : 0));

// 소프트맥스 회귀 784 → 15 를 페이지와 같은 순서로 계산합니다 (출력층 · 손실 · 예측 확률 블록,
// 경사 한 걸음, 값 바꾸기). batch = 1 이면 그림마다 갱신, 15 이면 15장 기울기를 더해 평균으로 갱신.
function srLogits(W, b, x, z) {
  for (let r = 0; r < QD.classes; r++) {
    let s = 0;
    for (let j = 0, base = r * QD.pixels; j < QD.pixels; j++) s += W[base + j] * x[j];
    z[r] = s;
  }
  for (let r = 0; r < QD.classes; r++) z[r] = z[r] + b[r];
}
function srSoftmax(z, p) {
  let m = -Infinity;
  for (const v of z) if (v > m) m = v;
  const e = new Float32Array(z.length);
  for (let r = 0; r < z.length; r++) e[r] = Math.exp(Math.fround(z[r] - m));
  let S = 0;
  for (let r = 0; r < z.length; r++) S += e[r];
  for (let r = 0; r < z.length; r++) p[r] = e[r] / S;
}
function srTrain({ lr, batch }) {
  const size = QD.classes * QD.pixels;
  const W = new Float32Array(size), b = new Float32Array(QD.classes);
  const gW = new Float32Array(size), gb = new Float32Array(QD.classes);
  const z = new Float32Array(QD.classes), p = new Float32Array(QD.classes);
  for (let i = 0; i < QD.train; i++) {
    const { x, label } = qdImage(i);
    srLogits(W, b, x, z); srSoftmax(z, p);
    for (let r = 0; r < QD.classes; r++) {
      const d = Math.fround(p[r] - (r === label ? 1 : 0));
      for (let j = 0, base = r * QD.pixels; j < QD.pixels; j++) gW[base + j] = gW[base + j] + Math.fround(d * x[j]);
      gb[r] = gb[r] + d;
    }
    if ((i + 1) % batch) continue;
    // 그림마다 갱신할 때는 평균 블록이 없습니다 (÷1 은 값이 같음).
    for (let q = 0; q < size; q++) { W[q] = W[q] - Math.fround(lr * Math.fround(gW[q] / batch)); gW[q] = 0; }
    for (let r = 0; r < QD.classes; r++) { b[r] = b[r] - Math.fround(lr * Math.fround(gb[r] / batch)); gb[r] = 0; }
  }
  return { W, b };
}
// 평가: 최고 확률이 여럿이면 정답이 그중에 있을 때 1 / 동률 수 (평가 패널과 같은 규칙).
function srEvaluate({ W, b }) {
  const z = new Float32Array(QD.classes), p = new Float32Array(QD.classes);
  const confusion = new Float64Array(QD.classes * QD.classes);
  let correct = 0;
  for (let i = QD.test.start; i < QD.test.start + QD.test.count; i++) {
    const { x, label } = qdImage(i);
    srLogits(W, b, x, z); srSoftmax(z, p);
    let m = -Infinity;
    for (const v of p) if (v > m) m = v;
    const winners = [];
    for (let r = 0; r < QD.classes; r++) if (p[r] === m) winners.push(r);
    if (winners.includes(label)) correct += 1 / winners.length;
    for (const r of winners) confusion[label * QD.classes + r] += 1 / winners.length;
  }
  const perClass = Array.from({ length: QD.classes }, (_, c) => confusion[c * QD.classes + c] / (QD.test.count / QD.classes));
  return { correct, accuracy: correct / QD.test.count, confusion, perClass };
}

const SR_ONLINE = { lr: 0.005, batch: 1 };
const SR_BATCH = { lr: 0.1, batch: 15 };
const MLP = { lr: 0.1, epochs: 5, hidden: 64, seed1: 5, seed2: 7, scale1: 0.0875, scale2: 0.25 };
// 17단계 MLP 는 흉내내기에 계산이 커서 페이지에서 잰 값을 적어 둡니다 (내용 검사가 다시 잽니다).
const MLP_MEASURED = { after: 0.7688666666666667 };
let srCache = null;
function srResults() {
  if (srCache) return srCache;
  const online = srTrain(SR_ONLINE), batched = srTrain(SR_BATCH);
  srCache = { online: srEvaluate(online), batched: { ...srEvaluate(batched), weights: batched } };
  return srCache;
}

// 데이터 블록 → 그림, 784 벡터, 종류 벡터
function sampleFront(ws, index, col, row) {
  const dataset = ws.add('dataset', {}, col, row + 1);
  const sample = ws.op('datasetSampleByIndex', [dataset, index], col + 1, row);
  const image = ws.op('sampleImage', [sample], col + 2, row);
  const x = ws.op('flatten', [image], col + 3, row);
  const y = ws.op('sampleOneHot', [sample], col + 3, row + 1);
  return { sample, image, x, y };
}

const qdMatrix = (name, rows, cols, extra = {}) => ({ mode: 'matrix', rows, cols, value: 0, ...extra });
const qdVector = (length, extra = {}) => ({ mode: 'vector', length, value: 0, ...extra });

// ---------- 11단계: 데이터 블록 ----------

function build11() {
  const ws = new Workspace();
  const dataset = ws.add('dataset', {}, 0, 0);
  [0, 1, 15, 15000].forEach((i, n) => {
    const R = n * 1.4;
    const index = ws.number(i, 0, R + 1);
    const sample = ws.op('datasetSampleByIndex', [dataset, index], 1, R);
    const image = ws.op('sampleImage', [sample], 2, R);
    const { x, label } = qdImage(i);
    ws.display(image, 3, R, { approx: Array.from(x), tol: 1e-7, shape: [28, 28] }, { view: 'image' });
    ws.display(ws.op('sampleLabel', [sample], 4, R), 5, R, { expect: label });
    ws.display(ws.op('sampleOneHot', [sample], 4, R + 0.7), 5, R + 0.7, { expect: oneHot(label) });
    if (n === 0) ws.display(ws.op('flatten', [image], 6, R), 7, R, { approx: Array.from(x), tol: 1e-7, shape: [784] });
  });
  return { ws, datasetSelection: QD_CLASSES, viewport: { x: 0, y: 0, zoom: 0.6 } };
}

// ---------- 12단계: Softmax ----------

const SOFTMAX_Z = [2, 1, 0.1];
const simSoftmax = z => { const m = Math.max(...z); const e = sim.exp(sim.sub(z, m)); return sim.div(e, sim.sum(e)); };

function build12() {
  const ws = new Workspace();
  const shifted = SOFTMAX_Z.map(v => v + 1000);
  const rows = [['z', SOFTMAX_Z, 0], ['z_큰값', shifted, 3]];
  for (const [name, values, R] of rows) {
    const z = ws.variable(name, qdVector(3), 0, R, values);
    const zf = Array.from(Float32Array.from(values));
    ws.display(z, 1, R, { approx: zf, tol: 1e-9 });
    // 그대로: e^z / Σe^z
    const e = ws.op('exp', [z], 1, R + 1);
    const naive = ws.op('divide', [e, ws.op('sum', [e], 2, R + 1.6)], 3, R + 1);
    ws.display(naive, 4, R + 1, R === 0 ? { approx: sim.div(sim.exp(zf), sim.sum(sim.exp(zf))), tol: 1e-7 } : null);
    // 안정화: 최댓값을 빼고 계산
    const p = ws.custom(CNN_BLOCKS.softmax15, [z], 2, R);
    ws.display(p, 3, R, { approx: simSoftmax(zf), tol: 1e-7 });
    ws.display(ws.op('sum', [p], 4, R), 5, R, { approx: 1, tol: 1e-6 });
  }
  return { ws, viewport: { x: 0, y: 0, zoom: 0.65 } };
}

// ---------- 13단계: 교차 엔트로피와 기울기 p − y ----------

const CE = { z: SOFTMAX_Z, label: 1, h: 0.01 };

function simCrossEntropy(z, y) {
  const m = Math.max(...z);
  const lse = m + Math.log(sim.sum(sim.exp(sim.sub(z, m))));
  return sim.sum(sim.mul(y, sim.sub(z, lse))) * -1;
}

function build13() {
  const ws = new Workspace();
  const yv = oneHot(CE.label, 3);
  const zf = Array.from(Float32Array.from(CE.z));
  const z = ws.variable('z', qdVector(3), 0, 0, CE.z);
  const y = ws.variable('y', qdVector(3), 0, 1, yv);
  const p = ws.custom(CNN_BLOCKS.softmax15, [z], 1, 0);
  const pv = simSoftmax(zf);
  ws.display(p, 2, 0, { approx: pv, tol: 1e-7 });
  const loss = ws.custom(CNN_BLOCKS.softmax_ce15, [z, y], 1, 1.2);
  ws.display(loss, 2, 1.2, { approx: simCrossEntropy(zf, yv), tol: 1e-6 });
  // 확인: −Σ y · log p
  const manual = ws.op('multiply', [ws.op('sum', [ws.op('multiply', [y, ws.op('log', [p], 2, 2.2)], 3, 2.2)], 4, 2.2), ws.number(-1, 4, 3)], 5, 2.2);
  ws.display(manual, 6, 2.2, { approx: -Math.log(pv[CE.label]), tol: 1e-5 });

  // 역전파: g = 1 → ∂z = p − y
  ws.variable('g', { value: 1 }, 0, 4);
  ws.custom(CNN_BLOCKS.softmax_ce15, [z, y], 1, 4, backwardParams2('g', ['dz', '']));
  const gradient = sim.sub(pv, yv);
  ws.display(ws.variable('dz', qdVector(3), 2, 4), 3, 4, { approx: gradient, tol: 1e-6 });

  // 수치 미분: z 의 k번째 칸만 ±h
  const h = ws.number(CE.h, 0, 6.5), twoH = ws.number(2 * CE.h, 0, 7.5);
  [0, 1, 2].forEach(k => {
    const R = 6.5 + k * 1.4;
    const ek = ws.variable(`e${k + 1}`, qdVector(3), 1, R, oneHot(k, 3));
    const step = ws.op('multiply', [ek, h], 2, R + 0.7);
    const up = ws.custom(CNN_BLOCKS.softmax_ce15, [ws.op('add', [z, step], 3, R), y], 4, R);
    const down = ws.custom(CNN_BLOCKS.softmax_ce15, [ws.op('subtract', [z, step], 3, R + 0.7), y], 4, R + 0.7);
    ws.display(ws.op('divide', [ws.op('subtract', [up, down], 5, R), twoH], 6, R), 7, R, { approx: gradient[k], tol: 1e-3 });
  });
  return { ws, viewport: { x: 0, y: 0, zoom: 0.55 } };
}

// ---------- 14·15단계: 소프트맥스 회귀 학습 ----------

// 역전파 변수와 순전파·역전파 블록. 반복 구조는 단계마다 다릅니다.
function softmaxRegression(ws, front, R, imageAt) {
  const Wparams = qdMatrix('W', QD.classes, QD.pixels, { trainable: 'yes' }), bparams = qdVector(QD.classes, { trainable: 'yes' });
  const W = ws.variable('W', Wparams, 4, R);
  const b = ws.variable('b', bparams, 4, R + 1);
  const z = ws.custom(CNN_BLOCKS.dense_linear15, [W, front.x, b], 5, R);
  const loss = ws.custom(CNN_BLOCKS.softmax_ce15, [z, front.y], 6, R);
  const probability = ws.custom(CNN_BLOCKS.softmax15, [z], 6, R + 1);
  ws.display(loss, 7, R);
  ws.display(probability, 7, R + 1);
  ws.display(front.image, ...imageAt, null, { view: 'image' });
  ws.variable('gL', { value: 1 }, 8.2, R);
  ws.variable('dz', qdVector(QD.classes), 8.2, R + 0.8);
  ws.variable('dW', qdMatrix('dW', QD.classes, QD.pixels), 9.2, R);
  ws.variable('db', qdVector(QD.classes), 9.2, R + 0.8);
  const backward = [
    ws.custom(CNN_BLOCKS.softmax_ce15, [z, front.y], 6, R + 2.5, backwardParams2('gL', ['dz', ''])),
    ws.custom(CNN_BLOCKS.dense_linear15, [W, front.x, b], 5, R + 2.5, backwardParams2('dz', ['dW', '', 'db']))
  ];
  return { W, b, Wparams, bparams, probability, backward };
}

// 학습한 W 의 k번째 줄을 28×28 로: 종류 k 의 점수를 올리는 픽셀(빨강)과 내리는 픽셀(파랑).
function weightTemplates(ws, W, col, row) {
  [0, 2, 5, 6, 7].forEach((k, n) => {
    const ek = ws.variable(`e_${QD_CLASSES[k]}`, qdVector(QD.classes), col, row + n * 1.2, oneHot(k));
    const line = ws.op('transposeMatvec', [W, ek], col + 1, row + n * 1.2);
    ws.display(ws.op('reshape', [line], col + 2, row + n * 1.2, { shape: '28,28' }), col + 3, row + n * 1.2, null, { view: 'image' });
  });
}

function build14() {
  const ws = new Workspace();
  const i = ws.variable('i', { value: 0 }, 0, 0);
  const front = sampleFront(ws, i, 0, 1);
  const model = softmaxRegression(ws, front, 0, [2, 2.3]);
  const eta = ws.number(SR_ONLINE.lr, 0, 4);
  const updates = [['W', 'dW'], ['b', 'db']].map(([name, grad], n) => {
    const g = ws.variable(grad, name === 'W' ? qdMatrix(grad, QD.classes, QD.pixels) : qdVector(QD.classes), 1 + n * 2, 5.5);
    return ws.op('setVariable', [ws.custom(SGD_STEP, [model[name], eta, g], 2 + n * 2, 5.5)], 2 + n * 2, 6.5, { variable: name });
  });
  const body = ws.op('sequence', [ws.chain(model.backward, 7, 3.5), ws.chain(updates, 7, 4.5)], 9, 4);
  ws.op('repeat', [body], 10, 4, { count: QD.train, start: 0, indexVariable: 'i' });
  const after = ws.variable('W', qdMatrix('W', QD.classes, QD.pixels, { trainable: 'yes' }), 0, 8);
  weightTemplates(ws, after, 1, 8);
  const expected = srResults().online;
  return {
    ws, datasetSelection: QD_CLASSES, viewport: { x: 0, y: 0, zoom: 0.5 },
    verify: { probabilityId: model.probability, ...QD.test, before: 1 / QD.classes, tol: 1e-9, train: true, after: expected.accuracy, afterTol: 0.002 }
  };
}

// 미니배치: 그림 15장(k)의 기울기를 g 에 모으고, 평균 g/15 로 한 번 갱신한 뒤 g ← 0.
// params: [{ name, shape, variable }] — variable 은 파라미터 변수 블록의 설정. 같은 이름의 변수 블록은
// 설정(서명)이 같아야 값을 공유하므로 갱신 줄의 파라미터 블록도 같은 설정으로 만듭니다.
function minibatchLoop(ws, params, backward, eta, col, row, epochs = 1) {
  const fifteen = ws.number(QD.classes, col, row + 3);
  const accumulate = [], update = [], reset = [];
  params.forEach(({ name, shape, variable }, n) => {
    const params = shape.length === 2 ? qdMatrix(name, shape[0], shape[1]) : qdVector(shape[0]);
    const local = ws.variable(`d${name}`, params, col + n * 2.2, row);
    const g = ws.variable(`g${name}`, params, col + n * 2.2, row + 1);
    accumulate.push(ws.op('setVariable', [ws.op('add', [g, local], col + 1 + n * 2.2, row)], col + 1 + n * 2.2, row + 0.8, { variable: `g${name}` }));
    const mean = ws.op('divide', [g, fifteen], col + 1 + n * 2.2, row + 1.8);
    update.push(ws.op('setVariable', [ws.custom(SGD_STEP, [ws.variable(name, variable, col + n * 2.2, row + 2), eta, mean], col + 1 + n * 2.2, row + 2.8)], col + 1 + n * 2.2, row + 3.6, { variable: name }));
    const zero = shape.length === 2
      ? ws.add('matrix', { rows: shape[0], cols: shape[1], init: 'constant', value: 0, seed: 1, scale: 0.05 }, col + n * 2.2, row + 4.4)
      : ws.add('constantVector', { length: shape[0], value: 0 }, col + n * 2.2, row + 4.4);
    reset.push(ws.op('setVariable', [zero], col + 1 + n * 2.2, row + 4.4, { variable: `g${name}` }));
  });
  const X = col + params.length * 2.2 + 0.5;
  const perSample = ws.op('sequence', [ws.chain(backward, X, row), ws.chain(accumulate, X, row + 0.8)], X + backward.length, row);
  const samples = ws.op('repeat', [perSample], X + backward.length + 1, row, { count: QD.classes, start: 0, indexVariable: 'k' });
  const perBatch = ws.op('sequence', [ws.op('sequence', [samples, ws.chain(update, X, row + 2)], X + backward.length + 2, row), ws.chain(reset, X, row + 3)], X + backward.length + 3, row);
  const batches = ws.op('repeat', [perBatch], X + backward.length + 4, row, { count: QD.train / QD.classes, start: 0, indexVariable: 'batch' });
  return epochs === 1 ? batches : ws.op('repeat', [batches], X + backward.length + 5, row, { count: epochs, start: 0, indexVariable: 'epoch' });
}

// 학습 그림 번호 = batch × 15 + k
function batchIndex(ws, col, row) {
  const batch = ws.variable('batch', { value: 0 }, col, row);
  const k = ws.variable('k', { value: 0 }, col, row + 1);
  return ws.op('add', [ws.op('multiply', [batch, ws.number(QD.classes, col, row + 2)], col + 1, row), k], col + 1, row + 1);
}

function build15() {
  const ws = new Workspace();
  const front = sampleFront(ws, batchIndex(ws, 0, 0), 0, 3);
  const model = softmaxRegression(ws, front, 0, [2, 4.4]);
  const eta = ws.number(SR_BATCH.lr, 0, 5.5);
  minibatchLoop(ws, [
    { name: 'W', shape: [QD.classes, QD.pixels], variable: model.Wparams }, { name: 'b', shape: [QD.classes], variable: model.bparams }
  ], model.backward, eta, 1, 7);
  const expected = srResults().batched;
  return {
    ws, datasetSelection: QD_CLASSES, viewport: { x: 0, y: 0, zoom: 0.45 },
    verify: { probabilityId: model.probability, ...QD.test, before: 1 / QD.classes, tol: 1e-9, train: true, after: expected.accuracy, afterTol: 0.002 }
  };
}

// ---------- 16단계: 평가를 블록으로 ----------

function build16() {
  const ws = new Workspace();
  const result = srResults().batched;
  const testI = ws.variable('test_i', { value: QD.test.start }, 0, 0);
  const front = sampleFront(ws, testI, 0, 1);
  const Wp = qdMatrix('W', QD.classes, QD.pixels, { trainable: 'yes' }), bp = qdVector(QD.classes, { trainable: 'yes' });
  const W = ws.variable('W', Wp, 4, 0), b = ws.variable('b', bp, 4, 1);
  ws.runtimeVariables.push(
    { name: 'W', signature: variableSignature(variableParams('W', Wp)), value: float32Value(result.weights.W, [QD.classes, QD.pixels]) },
    { name: 'b', signature: variableSignature(variableParams('b', bp)), value: float32Value(result.weights.b, [QD.classes]) }
  );
  const p = ws.custom(CNN_BLOCKS.softmax15, [ws.custom(CNN_BLOCKS.dense_linear15, [W, front.x, b], 5, 0)], 6, 0);
  ws.display(p, 7, 0);
  ws.display(front.image, 3, 3.2, null, { view: 'image' });

  // 예측 = 최고 확률 칸이 1 (동률이면 나눠 가짐). 정답 칸과 겹치는 만큼 맞음.
  const pred = ws.op('equal', [p, ws.op('arrayMax', [p], 6, 1.5)], 7, 1.5);
  const share = ws.op('divide', [pred, ws.op('sum', [pred], 7, 2.3)], 8, 1.5);
  const correct = ws.variable('맞은수', { value: 0 }, 9, 0);
  const confusion = ws.variable('혼동행렬', qdMatrix('혼동행렬', QD.classes, QD.classes), 9, 1);
  // 계산할 때마다 0부터 다시 셉니다. '계산'은 만든 순서대로 계산하므로 평가 반복보다 먼저 만듭니다.
  const resetCorrect = ws.op('setVariable', [ws.number(0, 11, 2.3)], 12, 2.3, { variable: '맞은수' });
  const resetConfusion = ws.op('setVariable', [ws.add('matrix', { rows: QD.classes, cols: QD.classes, init: 'constant', value: 0, seed: 1, scale: 0.05 }, 11, 3.1)], 12, 3.1, { variable: '혼동행렬' });
  const addCorrect = ws.op('setVariable', [ws.op('add', [correct, ws.op('dot', [front.y, share], 9, 2.3)], 10, 0)], 11, 0, { variable: '맞은수' });
  const addConfusion = ws.op('setVariable', [ws.op('add', [confusion, ws.op('outerProduct', [front.y, share], 9, 3.1)], 10, 1)], 11, 1, { variable: '혼동행렬' });
  const loop = ws.op('repeat', [ws.op('sequence', [addCorrect, addConfusion], 12, 0.5)], 13, 0.5, { count: QD.test.count, start: QD.test.start, indexVariable: 'test_i' });
  ws.op('sequence', [ws.op('sequence', [resetCorrect, resetConfusion], 13, 2.3), loop], 14, 1.5);

  // 결과: 반복 뒤에 만든 블록이라 평가가 끝난 값을 읽습니다.
  const Y = 5;
  const correctAfter = ws.variable('맞은수', { value: 0 }, 9, Y);
  ws.display(correctAfter, 10, Y, { approx: result.correct, tol: 1e-6 });
  ws.display(ws.op('divide', [correctAfter, ws.number(QD.test.count, 9, Y + 1)], 10, Y + 1), 11, Y + 1, { approx: result.accuracy, tol: 1e-9 });
  const confusionAfter = ws.variable('혼동행렬', qdMatrix('혼동행렬', QD.classes, QD.classes), 9, Y + 2);
  ws.display(confusionAfter, 10, Y + 2, { approx: Array.from(result.confusion), tol: 1e-6, shape: [QD.classes, QD.classes] }, { view: 'image' });
  // 종류별 정확도 = 혼동 행렬의 대각선 / 종류별 1,000장
  const eye = ws.variable('단위행렬', qdMatrix('단위행렬', QD.classes, QD.classes), 9, Y + 3.2, Array.from({ length: QD.classes * QD.classes }, (_, q) => (q % (QD.classes + 1) === 0 ? 1 : 0)));
  const ones = ws.add('constantVector', { length: QD.classes, value: 1 }, 10, Y + 4);
  const diagonal = ws.op('matvec', [ws.op('multiply', [confusionAfter, eye], 10, Y + 3.2), ones], 11, Y + 3.2);
  ws.display(ws.op('divide', [diagonal, ws.number(QD.test.count / QD.classes, 11, Y + 4)], 12, Y + 3.2), 13, Y + 3.2, { approx: result.perClass, tol: 1e-6 });
  return { ws, datasetSelection: QD_CLASSES, viewport: { x: 0, y: 0, zoom: 0.5 }, eval: result };
}

// ---------- 17단계: 은닉층을 더한 MLP ----------

function build17() {
  const ws = new Workspace();
  const front = sampleFront(ws, batchIndex(ws, 0, 0), 0, 3);
  const settings = {
    W1: qdMatrix('W1', MLP.hidden, QD.pixels, { init: 'random', seed: MLP.seed1, scale: MLP.scale1, trainable: 'yes' }),
    b1: qdVector(MLP.hidden, { trainable: 'yes' }),
    W2: qdMatrix('W2', QD.classes, MLP.hidden, { init: 'random', seed: MLP.seed2, scale: MLP.scale2, trainable: 'yes' }),
    b2: qdVector(QD.classes, { trainable: 'yes' })
  };
  const P = {
    W1: ws.variable('W1', settings.W1, 4, 0), b1: ws.variable('b1', settings.b1, 4, 1),
    W2: ws.variable('W2', settings.W2, 6, 0), b2: ws.variable('b2', settings.b2, 6, 1)
  };
  const h = ws.custom(CNN_BLOCKS.dense_relu64, [P.W1, front.x, P.b1], 5, 0.5);
  const z = ws.custom(CNN_BLOCKS.dense_linear15, [P.W2, h, P.b2], 7, 0.5);
  const loss = ws.custom(CNN_BLOCKS.softmax_ce15, [z, front.y], 8, 0);
  const probability = ws.custom(CNN_BLOCKS.softmax15, [z], 8, 1);
  ws.display(loss, 9, 0);
  ws.display(probability, 9, 1);
  ws.display(front.image, 3, 5.2, null, { view: 'image' });
  ws.variable('gL', { value: 1 }, 10, 0);
  ws.variable('dz', qdVector(QD.classes), 10, 0.8);
  ws.variable('dh', qdVector(MLP.hidden), 10, 1.6);
  const backward = [
    ws.custom(CNN_BLOCKS.softmax_ce15, [z, front.y], 8, 2.5, backwardParams2('gL', ['dz', ''])),
    ws.custom(CNN_BLOCKS.dense_linear15, [P.W2, h, P.b2], 7, 2.5, backwardParams2('dz', ['dW2', 'dh', 'db2'])),
    ws.custom(CNN_BLOCKS.dense_relu64, [P.W1, front.x, P.b1], 5, 2.5, backwardParams2('dh', ['dW1', '', 'db1']))
  ];
  const eta = ws.number(MLP.lr, 0, 6.5);
  minibatchLoop(ws, [
    { name: 'W1', shape: [MLP.hidden, QD.pixels], variable: settings.W1 }, { name: 'b1', shape: [MLP.hidden], variable: settings.b1 },
    { name: 'W2', shape: [QD.classes, MLP.hidden], variable: settings.W2 }, { name: 'b2', shape: [QD.classes], variable: settings.b2 }
  ], backward, eta, 1, 8, MLP.epochs);
  return {
    ws, datasetSelection: QD_CLASSES, viewport: { x: 0, y: 0, zoom: 0.45 },
    verify: { probabilityId: probability, ...QD.test, before: null, tol: 1, train: true, after: MLP_MEASURED.after, afterTol: 0.01 }
  };
}

// ---------- 단계 목록과 설명 ----------

function buildAll() {
  const s03 = build03();
  const s04 = build04();
  const s05 = build05(), s07 = build07(), s08 = build08(), s09 = build09(), s10 = build10();
  const s16 = build16();
  const steps = [
    { id: '00a', title: '0-1. 입력 블록과 값 보기', file: '00a-input-blocks.mmlab', build: build00a() },
    { id: '00b', title: '0-2. 수학 연산 블록', file: '00b-math-blocks.mmlab', build: build00b() },
    { id: '00c', title: '0-3. 값 바꾸기 · 둘 다 계산 · 반복', file: '00c-control-blocks.mmlab', build: build00c() },
    { id: '00d', title: '0-4. 사용자 블록', file: '00d-user-blocks.mmlab', build: build00d() },
    { id: '01', title: '1. 퍼셉트론: AND · OR · NAND', file: '01-perceptron.mmlab', build: build01() },
    { id: '02', title: '2. XOR 게이트 조립', file: '02-xor.mmlab', build: build02() },
    { id: '03', title: '3. 신경망의 기본 발상 Wx+b: XOR 학습 실패', file: '03-linear-fail.mmlab', build: s03 },
    { id: '04', title: '4. 임계값 활성화: 표현은 되지만 학습은?', file: '04-threshold-search.mmlab', build: s04 },
    { id: '05', title: '5. 경사하강: 기울기를 따라 내려가기', file: '05-gradient-descent.mmlab', build: s05 },
    { id: '06', title: '6. 매끄러운 활성화 함수: 시그모이드와 ReLU', file: '06-smooth-activation.mmlab', build: build06() },
    { id: '07', title: '7. 연쇄 법칙과 수동 역전파', file: '07-chain-rule.mmlab', build: s07 },
    { id: '08', title: '8. 뉴런 하나를 경사하강으로 학습: AND', file: '08-neuron-and.mmlab', build: s08 },
    { id: '09', title: '9. 2-2-1 신경망으로 XOR 학습 성공', file: '09-xor-backprop.mmlab', build: s09 },
    { id: '10', title: '10. 층 블록 Wx + b 와 그 역전파', file: '10-dense-layer.mmlab', build: s10 },
    { id: '11', title: '11. 데이터 블록: Quick Draw 그림과 정답', file: '11-data-blocks.mmlab', build: build11() },
    { id: '12', title: '12. Softmax: 점수를 확률로', file: '12-softmax.mmlab', build: build12() },
    { id: '13', title: '13. 교차 엔트로피와 기울기 p − y', file: '13-cross-entropy.mmlab', build: build13() },
    { id: '14', title: '14. 소프트맥스 회귀: 그림 한 장씩 학습', file: '14-softmax-regression.mmlab', build: build14() },
    { id: '15', title: '15. 미니배치: 15장의 기울기를 모아 한 번에', file: '15-minibatch.mmlab', build: build15() },
    { id: '16', title: '16. 평가: 정확도와 혼동 행렬', file: '16-evaluation.mmlab', build: s16 },
    { id: '17', title: '17. 은닉층을 더한 MLP', file: '17-mlp.mmlab', build: build17() },
    { id: 'cnn-train', title: 'CNN 학습: 무작위 가중치에서 1 epoch', file: 'cnn-1-train.mmlab', build: buildCnn({ trained: false }) },
    { id: 'cnn-trained', title: 'CNN 완성: 60 epoch 학습 결과 평가와 손그림 분류', file: 'cnn-2-trained.mmlab', build: buildCnn({ trained: true }) }
  ];

  const descriptions = {
    '00a': [
      '블록은 왼쪽 팔레트에서 추가하고, 출력 포트(오른쪽)에서 입력 포트(왼쪽)로 끌어 연결합니다.',
      '값을 확인하려면 ‘값 보기’ 블록에 연결하고 위쪽 ‘계산’을 누릅니다. 블록을 선택하면 오른쪽 인스펙터에서 설정을 바꿀 수 있습니다.',
      '',
      '왼쪽 열',
      '· 숫자: 상수 하나.',
      '· 변수: 이름이 있는 값. 숫자·벡터·행렬 형태를 고르고, 벡터·행렬은 같은 값 또는 고정 시드 무작위로 채웁니다. ‘값 바꾸기’로 바꿀 수 있는 유일한 블록이라 가중치는 모두 변수로 만듭니다.',
      '오른쪽 열',
      '· 같은 값 벡터 / 무작위 벡터 / 행렬 만들기: 이름 없는 값 생성기. 시드가 같으면 언제나 같은 무작위 값이 나옵니다.'
    ],
    '00b': [
      '모든 연산은 숫자와 배열을 원소별로 다룹니다. 숫자와 벡터를 더하면 모든 원소에 더해집니다(브로드캐스트).',
      '',
      '왼쪽 열: 더하기, 빼기, 곱하기, 나누기, 벡터 + 숫자, 벡터 × 벡터, 최댓값 max(a, b).',
      '오른쪽 열: 같음?, 합, 내적, 행렬 × 벡터, 지수, 로그, 배열 최댓값.',
      '',
      '확인 포인트',
      '· max(x, 0)은 뒤에서 ReLU 활성화 함수로 다시 등장합니다.',
      '· 같음?은 같으면 1, 다르면 0을 돌려줍니다. 조건문이 없는 이 도구에서 조건을 수식으로 표현하는 방법입니다.'
    ],
    '00c': [
      '학습은 ‘값을 조금 바꾸고, 반복한다’의 연속입니다. 그 두 가지를 맡는 블록입니다.',
      '',
      'A. 값 바꾸기 + 반복: n ← n + 1 을 10번 반복하면 n = 10.',
      'B. 반복 번호: 반복 블록의 ‘반복 번호 변수’(i)와 같은 이름의 변수 블록은 반복 안에서 현재 회차 번호가 됩니다. i² 을 1~5까지 누적하면 55.',
      'C. 둘 다 계산: 한 회차에 값 바꾸기를 여러 개 실행할 때 묶어 줍니다. a ← b, b ← a + b 를 10번 반복하면 피보나치 수 89, 144.',
      '   값 바꾸기는 회차가 끝날 때 동시에 반영되므로 b ← a + b 는 바뀌기 전의 a를 씁니다.',
      '',
      '주의: ‘계산’은 모든 블록을 만든 순서대로 한 번씩 계산합니다. 반복이 끝난 뒤의 값을 보려면 반복 블록보다 나중에 만든 변수 블록으로 읽어야 합니다(오른쪽 값 보기).'
    ],
    '00d': [
      '여러 블록을 선택해 ‘묶기’를 누르면 하나의 사용자 블록이 됩니다. 바깥에서 들어오던 연결은 입력 포트가 되고, 마지막 블록이 출력이 됩니다.',
      '',
      '· ReLU: max(x, 0) 한 블록을 묶은 것.',
      '· 제곱: x를 곱하기의 두 입력에 모두 넣은 것. 하나의 입력을 여러 곳에 쓰려면 안에서 값 보기 블록으로 갈라 씁니다.',
      '· 블록 안의 블록: ReLU 블록과 제곱 블록을 다시 묶은 것.',
      '· 맨 아래 max(−3, 0)은 발표 중 직접 묶어 보는 재료입니다.',
      '',
      '시연: 사용자 블록을 선택하고 ‘내부 블록 복사’ 또는 ‘내부 구조 편집’으로 안을 열어 보입니다. 팔레트의 ‘내 블록’에서 다시 꺼내 쓸 수 있습니다.'
    ],
    '01': [
      '퍼셉트론은 입력에 가중치를 곱해 더하고, 문턱을 넘으면 1, 아니면 0을 내는 가장 단순한 뉴런 모형입니다.',
      '  y = step(w₁x₁ + w₂x₂ + b),   step(z) = 1 (z > 0), 0 (그 외)',
      '',
      '입력 x₁ = [0,0,1,1], x₂ = [0,1,0,1]은 진리표 네 줄을 벡터 하나에 담은 것이라, 한 번 계산하면 네 경우가 모두 나옵니다.',
      '',
      '위: 곱하기·더하기·임계값 함수 블록으로 퍼셉트론을 직접 조립. 가중치 (1, 1, −1.5)이면 AND. 아래쪽 값 보기는 문턱을 넘기 전의 값 w₁x₁ + w₂x₂ + b 입니다.',
      '가운데: 같은 계산을 ‘퍼셉트론’ 사용자 블록 하나로 묶은 것.',
      '아래: 가중치까지 안에 넣어 게이트 블록으로 만든 것.',
      '  AND  (1, 1, −1.5) → [0,0,0,1]',
      '  OR   (1, 1, −0.5) → [0,1,1,1]',
      '  NAND (−1, −1, 1.5) → [1,1,1,0]',
      '',
      '시연: 게이트 블록의 ‘내부 블록 복사’로 안의 퍼셉트론과 가중치를 보이고, 숫자 하나를 바꿔 결과가 어떻게 달라지는지 확인합니다.'
    ],
    '02': [
      'XOR은 ‘둘 중 하나만 1’일 때 1: [0,1,1,0].',
      '퍼셉트론 하나로는 만들 수 없습니다. 평면에서 (0,1), (1,0)과 (0,0), (1,1)을 직선 하나로 나눌 수 없기 때문입니다.',
      '',
      '하지만 1단계의 게이트를 이어 붙이면 됩니다.',
      '  XOR(x₁, x₂) = AND( OR(x₁, x₂), NAND(x₁, x₂) )',
      '',
      '위: OR과 NAND의 출력을 AND에 넣어 XOR을 조립 → [0,1,1,0].',
      '아래: 그것을 다시 ‘XOR 게이트’ 블록으로 묶고, 정답 y와 비교해 맞은 개수를 셉니다 → 4.',
      '',
      '핵심: 뉴런(게이트)을 층으로 쌓으면 뉴런 하나로 못 하는 일을 할 수 있습니다. 다만 여기서는 가중치를 우리가 손으로 정했습니다. 다음 단계부터는 가중치를 스스로 찾게 합니다.'
    ],
    '03': [
      '가중치를 사람이 정하지 않고 데이터에서 찾게 하는 것이 학습입니다.',
      '가장 단순한 모형부터: 활성화 함수가 없는 ŷ = w₁x₁ + w₂x₂ + b.',
      '',
      '손실 L = Σ(ŷ − y)² 이 작아지도록 가중치를 조금씩 움직입니다.',
      '  e = ŷ − y',
      '  w₁ ← w₁ − η·Σ(e·x₁),   w₂ ← w₂ − η·Σ(e·x₂),   b ← b − η·Σe',
      '이 식은 퍼셉트론 학습 규칙(델타 규칙)과 같은 모양입니다. 오차 e에 입력 x를 곱한 만큼 가중치를 되돌립니다.',
      '',
      `위: 학습 전 예측과 손실(초깃값 w₁ = ${LINEAR_INIT.w1}, w₂ = ${LINEAR_INIT.w2}, b = ${LINEAR_INIT.b}).`,
      `가운데: 오차 → 기울기 → 경사 한 걸음 → 값 바꾸기 → 반복 ${LINEAR_STEPS}회 (η = ${LINEAR_LR}).`,
      '아래: 학습 후 가중치, 예측, 손실.',
      '',
      `결과: w₁, w₂ → 0, b → 0.5. 예측은 네 경우 모두 0.5가 되고 손실은 ${s03.sim.lossAfter.toFixed(2)}에서 멈춥니다.`,
      '직선(평면) 하나로는 XOR을 어떻게 기울여도 반씩밖에 맞출 수 없기 때문입니다. 선형 모형은 층을 더 쌓아도 결국 하나의 선형식이라 결과가 같습니다.'
    ],
    '04': [
      '2단계처럼 임계값 뉴런을 2층으로 쌓으면 XOR을 표현할 수는 있습니다.',
      '위: 은닉 뉴런 2개(OR, NAND 가중치) + 출력 뉴런 1개(AND 가중치) → [0,1,1,0]. 파라미터는 모두 9개.',
      '',
      '그러면 이 9개를 학습으로 찾을 수 있을까요? 3단계처럼 기울기를 쓰려 하면 막힙니다. 임계값 함수는 계단 모양이라 기울기가 어디서나 0이고, 손실(오답 개수)도 0, 1, 2, 3, 4 다섯 값밖에 없어 ‘조금 나아지는 방향’이 없습니다.',
      '',
      '그래서 여기서는 탐색으로 찾습니다.',
      `  1. 후보 목록(9 × ${SEARCH_N.toLocaleString()} 무작위 행렬)에서 i번째 후보 c를 꺼낸다.`,
      '  2. 현재 w와 후보 c로 각각 오답 개수를 센다.',
      '  3. 후보가 더 나쁘지 않으면 w ← c (채택), 아니면 유지.',
      `  4. ${SEARCH_N.toLocaleString()}번 반복. 아직 못 맞춘 회차 수를 ‘tries’에 센다.`,
      '',
      `아래: 찾은 w, 학습 후 예측 [0,1,1,0], 오답 개수 0, 그리고 정답을 처음 찾기까지 걸린 시도 횟수 ${s04.search.tries.toLocaleString()}.`,
      '',
      '핵심: 기울기 정보가 없으면 탐색은 결국 ‘찍기’입니다. 파라미터 9개는 천 번쯤 찍으면 되지만, 마지막 단계의 CNN은 파라미터가 157,015개입니다.',
      '매끄럽게 미분되는 활성화 함수와, 기울기를 따라 내려가는 방법(역전파)이 필요한 이유입니다.'
    ],
    '05': [
      '04단계는 ‘조금 나아지는 방향’이 없어서 막혔습니다. 기울기는 바로 그 방향을 알려 줍니다.',
      `손실 L(w) = (w − ${GD.target})² 의 기울기는 dL/dw = 2(w − ${GD.target}). 기울기의 반대쪽으로 조금씩 움직이면 L이 작아집니다.`,
      '  w ← w − η · dL/dw   (η: 학습률, 한 걸음의 크기)',
      '',
      `위: w = ${GD.start}에서 시작, η = ${GD.good}. 반복 ${GD.steps}회 → w = ${s05.good.w.toFixed(5)}. 3에 다가갑니다.`,
      `아래: 같은 식에 η = ${GD.bad}. 반복 ${GD.steps}회 → u = ${Math.round(s05.bad.w).toLocaleString()}. 한 걸음이 너무 커서 3을 지나쳐 반대편으로 더 멀리 튀고, 점점 발산합니다.`,
      '',
      `왜? 한 번 갱신하면 (w − 3)이 (1 − 2η)배가 됩니다. η = ${GD.good}이면 ${1 - 2 * GD.good}배씩 줄고, η = ${GD.bad}이면 ${round3(1 - 2 * GD.bad)}배씩 부호를 바꾸며 커집니다.`,
      '시연: η 숫자 블록을 0.5(한 번에 도착), 0.9(진동하며 수렴)로 바꿔 계산해 봅니다.'
    ],
    '06': [
      '경사하강에는 기울기가 필요합니다. 04단계의 계단 함수는 기울기가 거의 모든 곳에서 0이라 쓸 수 없었습니다.',
      '',
      '위: 계단 함수와 그 수치 미분 (f(x + h) − f(x − h)) / 2h. x = 0 한 점을 빼면 0입니다.',
      '가운데: 시그모이드 σ(x) = 1 / (1 + e⁻ˣ). 지수, 더하기, 나누기 블록으로 조립했습니다(블록 안을 열어 확인).',
      '  0과 1 사이를 매끄럽게 잇는 ‘부드러운 계단’이고, 기울기는 σ\'(x) = σ(x)(1 − σ(x)) 로 순전파 값만으로 계산됩니다.',
      `  오른쪽 끝 값 보기: 수치 미분과 σ(1 − σ)의 최대 차이 → 거의 0 (h = ${DIFF_H}).`,
      '아래: ReLU = max(x, 0). 기울기는 x > 0이면 1, 아니면 0, 즉 계단 함수입니다. 계산이 싸서 마지막 CNN에서 씁니다.',
      '',
      '그래프는 ‘그래프 그리기’ 블록이 만든 행렬을 값 보기의 그림 보기로 본 것입니다(가로 x: −6~6).'
    ],
    '07': [
      '합성 함수의 미분은 바깥 함수의 기울기 × 안쪽 함수의 기울기 (연쇄 법칙).',
      `  z = w × x,  y = σ(z)  →  ∂y/∂w = σ\'(z) · x,   ∂y/∂x = σ\'(z) · w`,
      '',
      '이 도구에는 자동 미분이 없습니다. 블록마다 ‘역전파 정의’를 직접 씁니다(블록 선택 → 인스펙터 ‘역전파 정의 편집’).',
      '정의 안에서 쓸 수 있는 값: 입력, 순전파 출력 y, 위에서 내려온 기울기 g = ∂L/∂y.',
      '  곱하기:   ∂a = g × b,   ∂b = g × a',
      '  시그모이드: ∂x = g × y(1 − y)',
      '',
      `위: w = ${CHAIN.w}, x = ${CHAIN.x} 일 때 순전파 z = ${s07.chain.zv}, y = ${s07.chain.yv.toFixed(4)}.`,
      '가운데: 역전파 실행. 블록을 ‘역전파 실행’으로 두면 위에서 내려온 기울기 변수를 읽어 입력별 기울기를 변수에 씁니다. 순서는 출력 쪽부터 직접 정합니다(둘 다 계산).',
      `  g = 1 → σ 역전파 → dz = ${s07.chain.slope.toFixed(4)} → 곱하기 역전파 → dw = ${(s07.chain.slope * CHAIN.x).toFixed(4)}, dx = ${(s07.chain.slope * CHAIN.w).toFixed(4)}`,
      `아래: 확인. w를 ±${CHAIN.h}만큼 움직인 수치 미분도 ${(s07.chain.slope * CHAIN.x).toFixed(4)}.`
    ],
    '08': [
      '이제 뉴런 하나(선형 모형 → 시그모이드)를 경사하강으로 학습합니다. 목표는 AND = [0,0,0,1].',
      '',
      '위: 순전파. 가중치가 모두 0이라 예측은 [0.5, 0.5, 0.5, 0.5], 손실 Σ(y − t)² = 1.',
      '가운데: 한 회차 = 역전파(손실 → 시그모이드 → 선형 모형) → 값 바꾸기 3개.',
      '  손실 역전파:  ∂y = 2(y − t)',
      '  선형 모형 역전파:  ∂w₁ = Σ g·x₁,  ∂w₂ = Σ g·x₂,  ∂b = Σ g   (네 경우의 기울기를 더함)',
      `  w ← w − η·∂w  (η = ${NEURON.lr}), 반복 ${NEURON.steps}회`,
      `아래: 학습 후 w₁ = ${s08.sim.w1.toFixed(2)}, w₂ = ${s08.sim.w2.toFixed(2)}, b = ${s08.sim.b.toFixed(2)}. 예측 [${s08.sim.after.map(v => v.toFixed(2)).join(', ')}], 손실 ${s08.sim.lossAfter.toFixed(4)}.`,
      '',
      '1단계에서 손으로 넣은 AND 가중치(1, 1, −1.5)와 비율이 비슷합니다. 이번에는 데이터에서 스스로 찾았습니다.'
    ],
    '09': [
      '04단계와 같은 2-2-1 구조(파라미터 9개)에서 계단 함수를 시그모이드로 바꿨습니다. 이제 기울기로 학습할 수 있습니다.',
      '',
      '위: 은닉 뉴런 h₁, h₂ → 출력 뉴런 y. 초깃값은 고정 시드 난수입니다.',
      '가운데: 역전파 7개를 출력 쪽부터 차례로. 출력 뉴런의 선형 모형 역전파가 ∂h₁ = g·v₁, ∂h₂ = g·v₂ 를 은닉 뉴런으로 넘겨 줍니다. 이것이 역전파(오차를 뒤로 전달)입니다.',
      `  η = ${XOR_NET.lr}, 반복 ${XOR_NET.steps.toLocaleString()}회`,
      `아래: 학습 후 예측 [${s09.sim.after.map(v => v.toFixed(2)).join(', ')}], 손실 ${s09.sim.lossAfter.toFixed(4)}.`,
      '',
      `03단계(선형 모형)는 손실 1.00에서 멈췄고, 04단계는 무작위 탐색이 필요했습니다. 은닉층 + 매끄러운 활성화 + 역전파로 XOR을 학습했습니다.`
    ],
    '10': [
      '09단계의 뉴런 두 개를 행렬 하나로 묶으면 ‘층’입니다:  h = σ(W₁x + b₁),  y = σ(W₂h + b₂).',
      '',
      '층 블록 Wx + b 의 역전파 (g = ∂L/∂(Wx + b)):',
      '  ∂W = g ⊗ x   (외적)',
      '  ∂x = Wᵀg      (전치 행렬 × 벡터, 앞 층으로 넘길 기울기)',
      '  ∂b = g',
      '한 번 정의해 두면 크기에 상관없이 씁니다. 마지막 CNN의 Dense 층들도 이 식 그대로입니다.',
      '',
      '위: 진리표 k번째 줄 x = X·[번호 = k] 를 하나씩 꺼내 순전파.',
      `가운데: 역전파 5개 → W, b 갱신. 줄 4개(k) × ${LAYER_NET.epochs.toLocaleString()}번(epoch). 09단계는 네 줄을 한꺼번에, 여기서는 한 줄씩 갱신합니다.`,
      `아래: 학습 후 네 줄의 예측 ${s10.sim.after.map(v => v.toFixed(2)).join(', ')}.`
    ],
    '11': [
      `이제 실제 데이터입니다. Quick Draw 손그림 ${QD.classes}종, 종류마다 28×28 흑백 그림.`,
      `  ${QD_CLASSES.join(', ')}`,
      '',
      '‘데이터 샘플’ 블록은 번호 i 하나로 그림을 고릅니다:  종류 = i mod 15,  그 종류의 ⌊i / 15⌋번째 그림.',
      '  i = 0 → 고양이 0번,  i = 1 → 물고기 0번,  i = 15 → 고양이 1번,  i = 15,000 → 고양이 1,000번',
      '‘그림값’은 픽셀을 0~1로 바꾼 28×28 배열(그림 보기), ‘펼치기’는 784칸 벡터, ‘종류 번호’와 ‘종류 벡터’(정답 칸만 1)는 정답입니다.',
      '',
      `학습용: 0 ~ ${(QD.train - 1).toLocaleString()} (종류마다 1,000장)`,
      `평가용: ${QD.test.start.toLocaleString()} ~ ${(QD.test.start + QD.test.count - 1).toLocaleString()} (학습에 한 번도 쓰지 않은 그림)`,
      '평가용을 따로 두는 이유: 외운 그림이 아니라 처음 보는 그림을 맞히는지 재야 하기 때문입니다.'
    ],
    '12': [
      `신경망의 마지막 출력은 종류마다 점수 하나(${QD.classes}개)입니다. 이 점수를 확률로 바꾸는 것이 Softmax:`,
      '  pₖ = e^(zₖ) / Σ e^(z)     (모두 양수, 합 1, 점수가 클수록 확률도 큼)',
      '',
      `위: z = [${SOFTMAX_Z.join(', ')}]. 식 그대로 계산해도, 블록으로 계산해도 같은 확률.`,
      `아래: z에 1,000을 더한 [${SOFTMAX_Z.map(v => v + 1000).join(', ')}]. 식 그대로 하면 e¹⁰⁰⁰이 컴퓨터가 담을 수 있는 수를 넘어 ∞ / ∞ = NaN.`,
      '',
      '해결: 가장 큰 점수를 먼저 빼고 계산합니다.  e^(zₖ − m) / Σ e^(z − m)',
      '분자와 분모에 똑같이 e⁻ᵐ이 곱해져 값은 그대로이고, 지수는 0 이하라 넘치지 않습니다. 그래서 아래 결과가 위와 같습니다.',
      '‘예측 · Softmax 확률’ 블록 안이 바로 이 계산이고, 마지막 CNN에서도 같은 블록을 씁니다.'
    ],
    '13': [
      '확률 p가 정답을 얼마나 잘 맞히는지 재는 손실이 교차 엔트로피:',
      '  L = −Σ yₖ · log pₖ = −log(정답 칸의 확률)',
      '정답 확률이 1이면 0, 0에 가까울수록 커집니다.',
      '',
      `위: z = [${SOFTMAX_Z.join(', ')}], 정답은 두 번째 칸. ‘손실 · Softmax Cross Entropy’ 블록과 −Σ y·log p 를 직접 계산한 값이 같습니다.`,
      '',
      '이 블록의 역전파는 아주 간단합니다:  ∂L/∂z = p − y',
      '  정답 칸: p − 1 < 0 → 그 점수를 올리고,  나머지 칸: p > 0 → 그 점수를 내립니다.',
      '가운데: 역전파 실행으로 구한 dz,  아래: z의 칸 하나씩 ±0.01 움직인 수치 미분. 세 칸 모두 같습니다.'
    ],
    '14': [
      `가장 단순한 분류기: 그림 784칸 → 점수 ${QD.classes}개.  z = Wx + b,  p = softmax(z).   (W: 15 × 784, 파라미터 ${(QD.classes * QD.pixels + QD.classes).toLocaleString()}개)`,
      '블록: 10단계의 층 블록과 같은 ‘출력층’, 13단계의 ‘손실’, 12단계의 ‘예측 확률’.',
      '',
      `한 회차 = 그림 i 하나 → 순전파 → 역전파(손실 → 출력층) → W, b 갱신 (η = ${SR_ONLINE.lr}).`,
      `반복: 학습용 그림 ${QD.train.toLocaleString()}장을 한 번씩 (1 epoch).`,
      '',
      '시연',
      `  1. ‘예측 확률’ 선택 → ‘그림별 예측 평가’(시작 15,000, 15,000장) → ${percent(1 / QD.classes)}. W가 모두 0이라 모든 점수가 같습니다.`,
      `  2. ‘계산’ (몇 초) → 다시 평가 → ${percent(srResults().online.accuracy)}.`,
      '  3. 아래 그림: 학습한 W의 줄 하나를 28×28로 펼친 것. 그 종류 점수를 올리는 픽셀이 빨강, 내리는 픽셀이 파랑입니다.',
      `     (${[0, 2, 5, 6, 7].map(k => QD_CLASSES[k]).join(', ')})`
    ],
    '15': [
      '그림 한 장마다 갱신하면 기울기가 그림마다 들쭉날쭉합니다. 여러 장의 기울기를 평균 내어 한 번에 갱신하는 것이 미니배치.',
      '',
      `학습 그림 번호 = batch × 15 + k.  k = 0~14 동안 기울기를 g에 더하고(누적), 15장이 끝나면  W ← W − η · g/15,  g ← 0.`,
      '번호를 이렇게 매기면 한 배치에 15종이 한 장씩 들어갑니다.',
      `반복: 15장(k) × 배치 ${(QD.train / QD.classes).toLocaleString()}개(batch) = 1 epoch,  η = ${SR_BATCH.lr}`,
      '',
      `결과: 테스트 정확도 ${percent(srResults().batched.accuracy)}. 14단계(${percent(srResults().online.accuracy)})와 거의 같은데 갱신 횟수는 1/15입니다.`,
      '마지막 CNN의 학습 반복도 정확히 이 구조(누적 → 갱신 → 초기화)입니다.'
    ],
    '16': [
      '평가도 블록으로 직접 셉니다. 15단계에서 학습한 W, b를 넣어 두었습니다.',
      '',
      `반복: 평가용 그림 ${QD.test.count.toLocaleString()}장(test_i = 15,000 ~ 29,999)마다`,
      '  예측 = [p = max(p)] (최고 확률 칸만 1, 동률이면 나눠 가짐)',
      '  맞은수 ← 맞은수 + 정답 · 예측',
      '  혼동행렬 ← 혼동행렬 + 정답 ⊗ 예측    (행: 정답 종류, 열: 예측 종류)',
      '',
      `결과: 맞은 수 ${Math.round(s16.eval.correct).toLocaleString()} / ${QD.test.count.toLocaleString()} = ${percent(s16.eval.accuracy)}.`,
      '혼동 행렬(그림 보기): 대각선이 진할수록 잘 맞힌 것, 대각선 밖의 칸은 어떤 종류를 무엇으로 헷갈렸는지입니다.',
      `종류별 정확도 = 대각선 / 1,000:  가장 높은 ${QD_CLASSES[s16.eval.perClass.indexOf(Math.max(...s16.eval.perClass))]} ${percent(Math.max(...s16.eval.perClass))},  가장 낮은 ${QD_CLASSES[s16.eval.perClass.indexOf(Math.min(...s16.eval.perClass))]} ${percent(Math.min(...s16.eval.perClass))}.`
    ],
    '17': [
      `출력층 앞에 은닉층(ReLU, ${MLP.hidden}개)을 하나 더합니다:  h = ReLU(W₁x + b₁),  z = W₂h + b₂.   (파라미터 ${(MLP.hidden * QD.pixels + MLP.hidden + QD.classes * MLP.hidden + QD.classes).toLocaleString()}개)`,
      '은닉층의 역전파: ∂h는 출력층 역전파가 Wᵀg로 넘겨 주고, ReLU는 z > 0인 칸만 기울기를 통과시킵니다.',
      `학습은 15단계와 같은 미니배치(η = ${MLP.lr}), ${MLP.epochs} epoch.`,
      '',
      '시연: ‘계산’ (수십 초) → ‘예측 확률’ 선택 → ‘그림별 예측 평가’',
      `결과: ${percent(MLP_MEASURED.after)}.  소프트맥스 회귀는 ${percent(srResults().batched.accuracy)}에서 더 학습해도 거의 오르지 않습니다(10 epoch 72.0%).`,
      '은닉층이 픽셀들의 조합(획, 모양)을 먼저 만들고 그 위에서 분류하기 때문입니다.',
      '그렇다면 그림의 ‘모양’을 처음부터 잘 찾는 층은 없을까? → 합성곱.'
    ],
    'cnn-train': [
      `Quick Draw 그림 15종을 분류하는 합성곱 신경망(CNN)입니다. 파라미터 ${CNN_PARAM_COUNT.toLocaleString()}개.`,
      '합성곱, ReLU, 평균 풀링, Dense, Softmax Cross Entropy 모두 일반 수학 블록을 묶은 사용자 블록이고, 역전파도 블록마다 직접 작성한 식입니다.',
      '',
      `위: 순전파. 그림 28×28 → 5×5 합성곱 ${CNN_BRANCHES}개(ReLU, 2×2 평균 풀링 → 각 144) → Dense 128 + ReLU → Dense 64 + ReLU → 15종 점수 → 손실, 예측 확률.`,
      '가운데: 역전파. 순전파 블록 바로 아래에 같은 블록을 ‘역전파 실행’으로 두고, 손실부터 거꾸로 기울기를 넘깁니다(dLogits → dh2 → dh1 → dP1~8). 오른쪽 ‘역전파 · 출력층부터 차례로’ 블록이 그 순서를 정합니다.',
      `아래: 파라미터 ${CNN_PARAMS.length}개를 한 줄에 하나씩.`,
      '  누적: 그림 15장 동안 g ← g + ∂L/∂p',
      '  갱신: 15장마다 v ← 0.9v + g/15 (모멘텀),  p ← p − η·v,  학습률 η = 0.02 / (1 + epoch/10)',
      '  초기화: g ← 0',
      '반복: 그림 15장(k) × 배치 1,000개(batch) × 1 epoch = 학습 그림 15,000장(0~14,999번).',
      '',
      '시연',
      `  1. ‘예측 확률’ 블록을 선택하고 오른쪽 ‘그림별 예측 평가’(시작 15,000, 15,000장)로 학습 전 정확도를 봅니다 → ${percent(CNN_MEASURED.initial)} (15종 찍기 수준).`,
      '  2. ‘계산’ → 1 epoch 학습 (수십 초).',
      `  3. 다시 평가 → ${percent(CNN_MEASURED.oneEpoch)}. 학습에 쓰지 않은 그림 15,000장으로 잰 정확도입니다.`,
      '값 보기는 ‘계산’을 시작할 때의 값입니다. 학습 후 값은 블록을 선택하고 ‘선택 계산’으로 봅니다.'
    ],
    'cnn-trained': [
      '같은 신경망을 60 epoch(그림 900,000장, 가중치 갱신 60,000번) 학습한 가중치를 넣어 둔 상태입니다. 발표 중에 돌리기에는 40분 가까이 걸려 미리 학습했습니다.',
      '',
      '시연',
      `  1. ‘예측 확률’ 블록을 선택하고 ‘그림별 예측 평가’ → 테스트 그림 15,000장 정확도 ${percent(CNN_MEASURED.trained)}.`,
      '     종류별 정확도와 ‘틀린 그림 랜덤 보기’로 어떤 그림을 헷갈리는지 봅니다.',
      '  2. 그림판에 직접 그리고 ‘손그림 분류’ → 15종 확률.',
      '',
      `정리: 1 epoch ${percent(CNN_MEASURED.oneEpoch)} → 60 epoch ${percent(CNN_MEASURED.trained)}. 가중치 ${CNN_PARAM_COUNT.toLocaleString()}개를 사람이 정하지 않고, 직접 만든 수학 블록과 역전파로 데이터에서 찾았습니다.`,
      '',
      '주의: 이 단계에서 ‘계산’을 누르면 학습률 0.02로 1 epoch를 더 학습해 가중치가 바뀝니다. 바뀌었다면 이전 → 다음으로 단계를 다시 불러오세요.'
    ]
  };

  return {
    steps: steps.map(step => ({
      id: step.id, title: step.title, file: step.file,
      project: step.build.ws.toProject(step.build.viewport, step.build.datasetSelection, step.build.builtins),
      checks: step.build.ws.checks,
      verify: step.build.verify
    })),
    descriptions: Object.fromEntries(Object.entries(descriptions).map(([id, lines]) => [id, lines.join('\n')])),
    notes: { linear: s03.sim, search: s04.search }
  };
}

function writeAll() {
  const all = buildAll();
  for (const entry of fs.readdirSync(HERE)) {
    if (entry.endsWith('.mmlab')) fs.unlinkSync(path.join(HERE, entry));
  }
  for (const step of all.steps) {
    fs.writeFileSync(path.join(HERE, step.file), JSON.stringify(step.project));
  }
  fs.writeFileSync(path.join(HERE, 'steps.json'), JSON.stringify({ steps: all.steps.map(({ id, title, file }) => ({ id, title, file })) }, null, 2) + '\n');
  fs.writeFileSync(path.join(HERE, 'descriptions.json'), JSON.stringify(all.descriptions, null, 2) + '\n');
  for (const step of all.steps) {
    const project = step.project;
    console.log(`${step.file.padEnd(28)} 노드 ${String(project.graph.nodes.length).padStart(3)} · 연결 ${String(project.graph.connections.length).padStart(3)} · 사용자 블록 ${project.userBlocks.length} · 검사 ${step.checks.length}`);
  }
  console.log(`3단계 학습 후 손실 ${all.notes.linear.lossAfter.toFixed(6)}, 예측 ${all.notes.linear.after.map(v => v.toFixed(4)).join(', ')}`);
  console.log(`4단계 후보 시드 ${all.notes.search.seed}, 시도 ${all.notes.search.tries}회, 학습 전 오답 ${all.notes.search.initialError}, 학습 후 오답 ${all.notes.search.finalError}`);
}

export { buildAll };

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) writeAll();
