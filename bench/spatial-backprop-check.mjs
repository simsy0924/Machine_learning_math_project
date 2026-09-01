#!/usr/bin/env node
// Check the hand-written backward of the CNN pattern this project actually uses:
// a 3x3 unfold feeding a filter, ReLU, a 2x2/stride-2 average pool, over four
// branches.
//
// The training-time unfold -> matvec fusion this file used to compare against
// was removed together with automatic differentiation, so there is no second
// implementation to diff. Instead the user-authored gradients are checked
// against central finite differences of the loss. The forward graph is built
// only from 슬라이딩 창 펼치기 / 행렬×벡터 / 배열 모양 바꾸기 / 최댓값 / 합, and the
// backward graph only from 슬라이딩 창 합치기 / 전치 행렬×벡터 / 외적 / 같음? /
// 곱하기 — all ordinary palette blocks.
//
// Away from the ReLU kink the loss is exactly linear in every parameter, so a
// central difference is not an approximation here: it is the same number the
// backward formula must produce. The data below keeps every pre-activation
// strictly positive so no perturbation can cross the kink; a separate case
// drives every pre-activation negative and requires the gradients to vanish,
// which is what exercises the mask.

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const CONTENT_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.bin': 'application/octet-stream'
};

function startServer() {
  const server = http.createServer((request, response) => {
    const requested = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    const relative = requested === '/' ? 'index.html' : requested.replace(/^\/+/, '');
    const filePath = path.resolve(ROOT, relative);
    if (filePath !== ROOT && !filePath.startsWith(ROOT + path.sep)) {
      response.writeHead(403).end('forbidden');
      return;
    }
    fs.readFile(filePath, (error, data) => {
      if (error) {
        response.writeHead(404).end('not found');
        return;
      }
      response.writeHead(200, { 'content-type': CONTENT_TYPES[path.extname(filePath)] || 'application/octet-stream' });
      response.end(data);
    });
  });
  return new Promise(resolve => {
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

function loadChromium() {
  const globalRoot = path.join(path.dirname(process.execPath), '..', 'lib', 'node_modules');
  const candidates = [process.env.PLAYWRIGHT_MODULE, 'playwright', path.join(globalRoot, 'playwright')].filter(Boolean);
  for (const candidate of candidates) {
    try { return require(candidate).chromium; }
    catch (error) { if (error.code !== 'MODULE_NOT_FOUND') throw error; }
  }
  throw new Error('playwright를 찾지 못했습니다. `npm i -g playwright` 또는 PLAYWRIGHT_MODULE 환경변수로 경로를 지정하세요.');
}

async function checkInPage() {
  function assert(condition, message) {
    if (!condition) throw new Error(message);
  }

  const BRANCH_ID = 'spatial-conv-relu-pool';
  const KERNEL = { kernelRows: 3, kernelCols: 3, strideRows: 1, strideCols: 1, padding: 0 };
  const POOL = { kernelRows: 2, kernelCols: 2, strideRows: 2, strideCols: 2, padding: 0 };
  const CONV_SHAPE = { shape: '26,26' };
  const FLAT_SHAPE = { shape: '676' };
  const AVG = { length: 4, value: 0.25 };
  const ONES = { length: 169, value: 1 };
  const source = manualBackpropInputSourceId;

  // One presentation branch as a 사용자 블록:
  //   A = unfold3(x) -> c = A·k -> S = reshape(c) -> R = max(0,S)
  //   P = unfold2(R) -> q = P·[¼,¼,¼,¼] -> L = Σq
  function installBranchBlock() {
    USER_BLOCKS.set(BRANCH_ID, {
      id: BRANCH_ID,
      name: 'Conv3×3 + ReLU + AvgPool',
      nodes: [
        { id: 1, type: 'unfold2d', params: { ...KERNEL } },
        { id: 2, type: 'matvec', params: {} },
        { id: 3, type: 'reshape', params: { ...CONV_SHAPE } },
        { id: 4, type: 'number', params: { value: 0 } },
        { id: 5, type: 'maximum', params: {} },
        { id: 6, type: 'unfold2d', params: { ...POOL } },
        { id: 7, type: 'constantVector', params: { ...AVG } },
        { id: 8, type: 'matvec', params: {} },
        { id: 9, type: 'sum', params: {} }
      ],
      connections: [
        { from: 1, to: 2, inputIndex: 0 },
        { from: 2, to: 3, inputIndex: 0 },
        { from: 4, to: 5, inputIndex: 0 },
        { from: 3, to: 5, inputIndex: 1 },
        { from: 5, to: 6, inputIndex: 0 },
        { from: 6, to: 8, inputIndex: 0 },
        { from: 7, to: 8, inputIndex: 1 },
        { from: 8, to: 9, inputIndex: 0 }
      ],
      externalInputs: [
        { nodeId: 1, inputIndex: 0, label: 'x' },
        { nodeId: 2, inputIndex: 1, label: 'k' }
      ],
      outputNodeId: 9,
      formula: 'Σ AvgPool(ReLU(conv(x, k)))',
      // Derived by hand:
      //   dq = g·1        dP = dq ⊗ avg      dR = fold2(dP, R)
      //   dS = dR·[R = S] dc = reshape(dS)   dk = Aᵀdc
      //   dA = dc ⊗ k     dx = fold3(dA, x)
      //
      // Nodes 5..9 below repeat the forward expressions exactly, so the
      // executor serves A, c, S and R out of the forward trace instead of
      // recomputing them.
      manualBackprop: {
        nodes: [
          { id: 1, type: 'constantVector', params: { ...ONES } },
          { id: 2, type: 'multiply', params: {} },
          { id: 3, type: 'constantVector', params: { ...AVG } },
          { id: 4, type: 'outerProduct', params: {} },
          { id: 5, type: 'unfold2d', params: { ...KERNEL } },
          { id: 6, type: 'matvec', params: {} },
          { id: 7, type: 'reshape', params: { ...CONV_SHAPE } },
          { id: 8, type: 'number', params: { value: 0 } },
          { id: 9, type: 'maximum', params: {} },
          { id: 10, type: 'fold2d', params: { ...POOL } },
          { id: 11, type: 'equal', params: {} },
          { id: 12, type: 'multiply', params: {} },
          { id: 13, type: 'reshape', params: { ...FLAT_SHAPE } },
          { id: 14, type: 'transposeMatvec', params: {} },
          { id: 15, type: 'outerProduct', params: {} },
          { id: 16, type: 'fold2d', params: { ...KERNEL } }
        ],
        connections: [
          { from: MANUAL_BACKPROP_UPSTREAM_ID, to: 2, inputIndex: 0 },
          { from: 1, to: 2, inputIndex: 1 },
          { from: 2, to: 4, inputIndex: 0 },
          { from: 3, to: 4, inputIndex: 1 },

          { from: source(0), to: 5, inputIndex: 0 },
          { from: 5, to: 6, inputIndex: 0 },
          { from: source(1), to: 6, inputIndex: 1 },
          { from: 6, to: 7, inputIndex: 0 },
          { from: 8, to: 9, inputIndex: 0 },
          { from: 7, to: 9, inputIndex: 1 },

          { from: 4, to: 10, inputIndex: 0 },
          { from: 9, to: 10, inputIndex: 1 },
          { from: 9, to: 11, inputIndex: 0 },
          { from: 7, to: 11, inputIndex: 1 },
          { from: 10, to: 12, inputIndex: 0 },
          { from: 11, to: 12, inputIndex: 1 },
          { from: 12, to: 13, inputIndex: 0 },

          { from: 5, to: 14, inputIndex: 0 },
          { from: 13, to: 14, inputIndex: 1 },
          { from: 13, to: 15, inputIndex: 0 },
          { from: source(1), to: 15, inputIndex: 1 },
          { from: 15, to: 16, inputIndex: 0 },
          { from: source(0), to: 16, inputIndex: 1 }
        ],
        gradientOutputNodeIds: [16, 14]
      }
    });
  }

  function add(type, params = null) {
    const node = addBlock(type);
    if (params) Object.assign(node.params, params);
    return node;
  }

  function link(from, to, inputIndex = 0) {
    graph.connections.push({ from: from.id, to: to.id, inputIndex });
  }

  function buildGraph() {
    resetWorkspace();
    installBranchBlock();

    const x = add('variable', { name: 'x', mode: 'matrix', rows: 28, cols: 28, init: 'constant', value: 0 });
    add('variable', { name: 'gL', mode: 'scalar', value: 1 });

    const branches = [];
    for (let filter = 0; filter < 4; filter++) {
      const k = add('variable', { name: `k${filter + 1}`, mode: 'vector', length: 9, init: 'constant', value: 0 });
      const branch = add(`custom:${BRANCH_ID}`);
      link(x, branch, 0);
      link(k, branch, 1);
      branches.push({ k, branch });
      add('variable', { name: `gk${filter + 1}`, mode: 'scalar', value: 0 });
      add('variable', { name: `gx${filter + 1}`, mode: 'scalar', value: 0 });
    }

    let loss = branches[0].branch;
    for (let i = 1; i < branches.length; i++) {
      const next = add('add');
      link(loss, next, 0);
      link(branches[i].branch, next, 1);
      loss = next;
    }

    // Every branch loss enters the total with weight 1, so each branch's
    // upstream gradient is the same gL.
    const backward = [];
    for (let filter = 0; filter < 4; filter++) {
      const node = add(`custom:${BRANCH_ID}`, {
        manualBackpropMode: 'backward',
        manualBackpropUpstream: 'gL',
        manualBackpropGradient0: `gx${filter + 1}`,
        manualBackpropGradient1: `gk${filter + 1}`
      });
      link(x, node, 0);
      link(branches[filter].k, node, 1);
      backward.push(node);
    }

    // dL/dx is the sum of the four branch contributions.
    let total = add('variable', { name: 'gx1', mode: 'scalar', value: 0 });
    for (let filter = 1; filter < 4; filter++) {
      const part = add('variable', { name: `gx${filter + 1}`, mode: 'scalar', value: 0 });
      const next = add('add');
      link(total, next, 0);
      link(part, next, 1);
      total = next;
    }
    add('variable', { name: 'gx', mode: 'scalar', value: 0 });
    const storeGx = add('setVariable', { variable: 'gx' });
    link(total, storeGx, 0);

    let body = loss;
    for (const node of [...backward, storeGx]) {
      const sequence = add('sequence');
      link(body, sequence, 0);
      link(node, sequence, 1);
      body = sequence;
    }

    return { loss, body };
  }

  function inputData() {
    const data = new Float32Array(28 * 28);
    for (let i = 0; i < data.length; i++) data[i] = 0.2 + ((i * 37) % 101) / 125;
    return arrayValue(data, [28, 28]);
  }

  // Strictly positive so every pre-activation stays positive and the loss is
  // exactly linear in x and k.
  function kernelData(filter, sign = 1) {
    const data = new Float32Array(9);
    for (let i = 0; i < 9; i++) data[i] = sign * ((filter + 1) * 0.03 + (i + 1) * 0.01);
    return arrayValue(data, [9]);
  }

  function writeInputs(sign) {
    writeRuntimeVariable('x', inputData(), true);
    for (let filter = 0; filter < 4; filter++) {
      writeRuntimeVariable(`k${filter + 1}`, kernelData(filter, sign), true);
    }
  }

  function lossValue(lossId) {
    return evaluateNode(lossId, new Map(), new Set());
  }

  // Central difference on one element of a runtime variable. The loss is
  // piecewise linear, so away from the kink this is exact.
  function numericGradient(lossId, name, index, h) {
    const stored = asArrayValue(RUNTIME_VARIABLES.get(name).value);
    const original = stored.data[index];

    const probe = new Float32Array(stored.data);
    probe[index] = original + h;
    writeRuntimeVariable(name, arrayValue(probe, [...stored.shape]), true);
    const plus = lossValue(lossId);

    const probeMinus = new Float32Array(stored.data);
    probeMinus[index] = original - h;
    writeRuntimeVariable(name, arrayValue(probeMinus, [...stored.shape]), true);
    const minus = lossValue(lossId);

    writeRuntimeVariable(name, arrayValue(new Float32Array(stored.data), [...stored.shape]), true);
    return (plus - minus) / (2 * h);
  }

  function relativeError(actual, expected) {
    const scale = Math.max(1e-6, Math.abs(expected), Math.abs(actual));
    return Math.abs(actual - expected) / scale;
  }

  async function runGradients(built) {
    await evaluateGraph();
    const failed = [...graph.nodes.values()].find(node => node.lastError);
    if (failed) throw new Error(`${failed.type}: ${failed.lastError}`);
  }

  assert(BLOCKS.fold2d, 'fold2d 블록이 등록되지 않았습니다.');
  assert(BLOCKS.transposeMatvec, 'transposeMatvec 블록이 등록되지 않았습니다.');

  const built = buildGraph();

  // ---- 1. positive kernels: gradients must match finite differences ----
  writeInputs(1);
  await runGradients(built);

  const plan = manualBackpropExecutionPlanFor(
    USER_BLOCKS.get(BRANCH_ID).manualBackprop,
    2,
    [true, true],
    USER_BLOCKS.get(BRANCH_ID)
  );
  const reusedForwardSteps = plan.steps.filter(step => step.kind === 'forwardCache').length;

  const h = 0.05;
  let worstKernel = 0;
  for (let filter = 0; filter < 4; filter++) {
    const analytic = asArrayValue(RUNTIME_VARIABLES.get(`gk${filter + 1}`).value);
    assert(analytic.data.length === 9, `gk${filter + 1} 크기가 9가 아닙니다.`);
    for (let i = 0; i < 9; i++) {
      const numeric = numericGradient(built.loss.id, `k${filter + 1}`, i, h);
      worstKernel = Math.max(worstKernel, relativeError(analytic.data[i], numeric));
    }
  }
  assert(worstKernel < 2e-3, `커널 gradient가 수치 미분과 다릅니다. 최대 상대오차 ${worstKernel}`);

  const gx = asArrayValue(RUNTIME_VARIABLES.get('gx').value);
  assert(gx.shape[0] === 28 && gx.shape[1] === 28, '입력 gradient 모양이 28×28이 아닙니다.');
  let worstInput = 0;
  // Sample positions spread over the image, including the padding-free corners
  // where fewer 3×3 windows overlap.
  const probes = [0, 1, 27, 28, 29, 57, 200, 201, 391, 392, 500, 700, 754, 755, 782, 783];
  for (const index of probes) {
    const numeric = numericGradient(built.loss.id, 'x', index, h);
    worstInput = Math.max(worstInput, relativeError(gx.data[index], numeric));
  }
  assert(worstInput < 2e-3, `입력 gradient가 수치 미분과 다릅니다. 최대 상대오차 ${worstInput}`);

  // ---- 2. negative kernels: ReLU masks everything, gradients must vanish ----
  writeInputs(-1);
  await runGradients(built);
  for (let filter = 0; filter < 4; filter++) {
    const masked = asArrayValue(RUNTIME_VARIABLES.get(`gk${filter + 1}`).value);
    for (let i = 0; i < 9; i++) {
      assert(masked.data[i] === 0, `ReLU가 전부 막힌 가지의 gk${filter + 1}[${i}]가 0이 아닙니다.`);
    }
  }
  const maskedInput = asArrayValue(RUNTIME_VARIABLES.get('gx').value);
  for (let i = 0; i < maskedInput.data.length; i++) {
    assert(maskedInput.data[i] === 0, `ReLU가 전부 막힌 입력 gradient gx[${i}]가 0이 아닙니다.`);
  }

  // ---- 3. timing of one full manual gradient pass ----
  writeInputs(1);
  const iterations = 40;
  for (let i = 0; i < 8; i++) await runGradients(built);
  const started = performance.now();
  for (let i = 0; i < iterations; i++) await runGradients(built);
  const msPerPass = (performance.now() - started) / iterations;

  return {
    kernelGradients: 'finite-difference OK',
    inputGradients: 'finite-difference OK',
    reluMask: 'OK',
    worstKernelRelativeError: worstKernel,
    worstInputRelativeError: worstInput,
    reusedForwardSteps,
    msPerPass
  };
}

const chromium = loadChromium();
const { server, port } = await startServer();
let browser;
try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });
  if (pageErrors.length) throw new Error(`페이지 로딩 오류: ${pageErrors.join(' | ')}`);
  const result = await page.evaluate(checkInPage);
  console.log('spatial manual backprop checks:', result);
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
