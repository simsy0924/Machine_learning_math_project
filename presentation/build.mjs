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

function defineBlock({ id, name, formula, uses = [], build }) {
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
  return {
    id, uses,
    definition: { id, name, nodes: b.nodes, connections: b.connections, externalInputs: b.externalInputs, outputNodeId: b.outputNodeId, formula }
  };
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

const LINEAR = defineBlock({
  id: 'linear2', name: '선형 모형 · w₁x₁ + w₂x₂ + b', formula: 'ŷ = w₁x₁ + w₂x₂ + b',
  build(b) { b.output(weightedSum(b)); }
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

const SQ_ERR = defineBlock({
  id: 'sq_err', name: '손실 · Σ(ŷ − y)²', formula: 'Σ(ŷ − y)²',
  build(b) {
    const e = b.add('subtract', {}, 0, 0);
    const sq = b.add('multiply', {}, 1, 0); b.link(e, sq, 0); b.link(e, sq, 1);
    const out = b.add('sum', {}, 2, 0); b.link(sq, out, 0);
    b.input(e, 0, 'ŷ'); b.input(e, 1, 'y'); b.output(out);
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
  display(from, col, row, check = null) {
    const id = this.add('display', {}, col, row);
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
  custom(block, inputs, col, row) {
    this.use(block);
    const id = this.add(`custom:${block.id}`, {}, col, row);
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
  toProject(viewport) {
    return {
      format: FORMAT, version: VERSION,
      graph: { nodes: this.nodes, connections: this.connections },
      userBlocks: Array.from(this.userBlocks.values()),
      runtimeVariables: this.runtimeVariables,
      datasetSelection: [],
      viewport
    };
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

// ---------- 단계 목록과 설명 ----------

function buildAll() {
  const s03 = build03();
  const s04 = build04();
  const steps = [
    { id: '00a', title: '0-1. 입력 블록과 값 보기', file: '00a-input-blocks.mmlab', build: build00a() },
    { id: '00b', title: '0-2. 수학 연산 블록', file: '00b-math-blocks.mmlab', build: build00b() },
    { id: '00c', title: '0-3. 값 바꾸기 · 둘 다 계산 · 반복', file: '00c-control-blocks.mmlab', build: build00c() },
    { id: '00d', title: '0-4. 사용자 블록', file: '00d-user-blocks.mmlab', build: build00d() },
    { id: '01', title: '1. 퍼셉트론: AND · OR · NAND', file: '01-perceptron.mmlab', build: build01() },
    { id: '02', title: '2. XOR 게이트 조립', file: '02-xor.mmlab', build: build02() },
    { id: '03', title: '3. 신경망의 기본 발상 Wx+b: XOR 학습 실패', file: '03-linear-fail.mmlab', build: s03 },
    { id: '04', title: '4. 임계값 활성화: 표현은 되지만 학습은?', file: '04-threshold-search.mmlab', build: s04 }
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
    ]
  };

  return {
    steps: steps.map(step => ({
      id: step.id, title: step.title, file: step.file,
      project: step.build.ws.toProject(step.build.viewport),
      checks: step.build.ws.checks
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
