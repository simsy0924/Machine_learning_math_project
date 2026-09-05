#!/usr/bin/env node
// Correctness checks for reshape + sliding-window unfold using the real browser app.
//
// There is no automatic differentiation, so the backward side is checked the
// way a user writes it: 슬라이딩 창 합치기 (fold2d) is the hand-written backward
// of 슬라이딩 창 펼치기, and the graph check below runs a full manual backward
// chain through the real 역전파 실행 path.

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
  const candidates = [
    process.env.PLAYWRIGHT_MODULE,
    'playwright',
    path.join(globalRoot, 'playwright')
  ].filter(Boolean);

  for (const candidate of candidates) {
    try {
      return require(candidate).chromium;
    } catch (error) {
      if (error.code !== 'MODULE_NOT_FOUND') throw error;
    }
  }

  throw new Error(
    'playwright를 찾지 못했습니다. `npm i -g playwright` 또는 PLAYWRIGHT_MODULE 환경변수로 경로를 지정하세요.'
  );
}

async function checkInPage() {
  function assert(condition, message) {
    if (!condition) throw new Error(message);
  }

  function arraysEqual(actual, expected, epsilon = 1e-6) {
    if (actual.length !== expected.length) return false;
    for (let i = 0; i < actual.length; i++) {
      if (Math.abs(actual[i] - expected[i]) > epsilon) return false;
    }
    return true;
  }

  assert(BLOCKS.reshape, 'reshape 블록이 등록되지 않았습니다.');
  assert(BLOCKS.unfold2d, 'unfold2d 블록이 등록되지 않았습니다.');
  assert(document.querySelector('[data-block="reshape"]'), 'reshape 팔레트 버튼이 없습니다.');
  assert(document.querySelector('[data-block="unfold2d"]'), 'unfold2d 팔레트 버튼이 없습니다.');

  const source = arrayValue(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
  const reshapeNode = { params: { shape: '3,2' } };
  const reshaped = BLOCKS.reshape.compute(reshapeNode, [source]);
  assert(reshaped.shape[0] === 3 && reshaped.shape[1] === 2, 'reshape 3×2 모양이 틀렸습니다.');
  assert(reshaped.data === source.data, 'reshape가 불필요하게 데이터 버퍼를 복사했습니다.');

  reshapeNode.params.shape = '-1,3';
  const inferred = BLOCKS.reshape.compute(reshapeNode, [source]);
  assert(inferred.shape[0] === 2 && inferred.shape[1] === 3, 'reshape -1 자동 계산이 틀렸습니다.');

  const image = arrayValue(new Float32Array([
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  ]), [3, 3]);
  const unfoldNode = {
    params: { kernelRows: 2, kernelCols: 2, strideRows: 1, strideCols: 1, padding: 0 }
  };
  const unfolded = BLOCKS.unfold2d.compute(unfoldNode, [image]);
  assert(unfolded.shape[0] === 4 && unfolded.shape[1] === 4, '2×2 unfold 출력 모양이 틀렸습니다.');
  assert(arraysEqual(Array.from(unfolded.data), [
    1, 2, 4, 5,
    2, 3, 5, 6,
    4, 5, 7, 8,
    5, 6, 8, 9
  ]), '2×2 unfold 값이 틀렸습니다.');

  // fold2d is what a user writes as the backward of unfold2d: every patch value
  // goes back to the position it came from and overlapping windows are summed.
  assert(BLOCKS.fold2d, 'fold2d 블록이 등록되지 않았습니다.');
  assert(document.querySelector('[data-block="fold2d"]'), 'fold2d 팔레트 버튼이 없습니다.');
  const ones = arrayValue(new Float32Array(16).fill(1), [4, 4]);
  const foldNode = {
    params: { kernelRows: 2, kernelCols: 2, strideRows: 1, strideCols: 1, padding: 0 }
  };
  const inputGradient = BLOCKS.fold2d.compute(foldNode, [ones, image]);
  assert(inputGradient.shape[0] === 3 && inputGradient.shape[1] === 3, 'fold2d 출력 모양이 틀렸습니다.');
  assert(arraysEqual(Array.from(inputGradient.data), [
    1, 2, 1,
    2, 4, 2,
    1, 2, 1
  ]), '겹치는 창의 unfold gradient 누적이 틀렸습니다.');

  unfoldNode.params = { kernelRows: 3, kernelCols: 3, strideRows: 1, strideCols: 1, padding: 1 };
  const padded = BLOCKS.unfold2d.compute(unfoldNode, [image]);
  assert(padded.shape[0] === 9 && padded.shape[1] === 9, 'padding unfold 출력 모양이 틀렸습니다.');
  assert(arraysEqual(Array.from(padded.data.slice(0, 9)), [0, 0, 0, 0, 1, 2, 0, 4, 5]), 'padding 값이 틀렸습니다.');

  // Full graph check: L = sum(unfold(x) @ k), with the backward chain written
  // out of ordinary blocks and executed through 역전파 실행, exactly as it would
  // be in the workspace.
  resetWorkspace();
  const source0 = manualBackpropInputSourceId(0);
  const source1 = manualBackpropInputSourceId(1);

  // dL/dx = g·1 for every element. equal(x, x) is the ones array of x's shape.
  MANUAL_BACKPROP_BUILTINS.set('sum', {
    nodes: [
      { id: 1, type: 'equal', params: {} },
      { id: 2, type: 'multiply', params: {} }
    ],
    connections: [
      { from: source0, to: 1, inputIndex: 0 },
      { from: source0, to: 1, inputIndex: 1 },
      { from: MANUAL_BACKPROP_UPSTREAM_ID, to: 2, inputIndex: 0 },
      { from: 1, to: 2, inputIndex: 1 }
    ],
    gradientOutputNodeIds: [2]
  });

  // y = Ax -> dA = g ⊗ x, dx = Aᵀg
  MANUAL_BACKPROP_BUILTINS.set('matvec', {
    nodes: [
      { id: 1, type: 'outerProduct', params: {} },
      { id: 2, type: 'transposeMatvec', params: {} }
    ],
    connections: [
      { from: MANUAL_BACKPROP_UPSTREAM_ID, to: 1, inputIndex: 0 },
      { from: source1, to: 1, inputIndex: 1 },
      { from: source0, to: 2, inputIndex: 0 },
      { from: MANUAL_BACKPROP_UPSTREAM_ID, to: 2, inputIndex: 1 }
    ],
    gradientOutputNodeIds: [1, 2]
  });

  // y = unfold(x) -> dx = fold(g, x), with the same window geometry.
  MANUAL_BACKPROP_BUILTINS.set('unfold2d', {
    nodes: [{
      id: 1,
      type: 'fold2d',
      params: { kernelRows: 2, kernelCols: 2, strideRows: 1, strideCols: 1, padding: 0 }
    }],
    connections: [
      { from: MANUAL_BACKPROP_UPSTREAM_ID, to: 1, inputIndex: 0 },
      { from: source0, to: 1, inputIndex: 1 }
    ],
    gradientOutputNodeIds: [1]
  });

  const make = (type, params) => {
    const node = addBlock(type);
    if (params) Object.assign(node.params, params);
    return node;
  };
  const wire = (from, to, inputIndex = 0) => {
    graph.connections.push({ from: from.id, to: to.id, inputIndex });
  };

  const x = make('variable', { name: 'x', mode: 'matrix', rows: 3, cols: 3, init: 'constant', value: 0 });
  const k = make('variable', { name: 'k', mode: 'vector', length: 4, init: 'constant', value: 0 });
  const unfold = make('unfold2d', { kernelRows: 2, kernelCols: 2, strideRows: 1, strideCols: 1, padding: 0 });
  wire(x, unfold, 0);
  const matvec = make('matvec');
  wire(unfold, matvec, 0);
  wire(k, matvec, 1);
  const loss = make('sum');
  wire(matvec, loss, 0);

  make('variable', { name: 'gL', mode: 'scalar', value: 1 });
  for (const name of ['gm', 'gA', 'gk', 'gx']) {
    make('variable', { name, mode: 'scalar', value: 0 });
  }

  const backwardBlock = (type, params, inputs) => {
    const node = make(type, { manualBackpropMode: 'backward', ...params });
    inputs.forEach((from, inputIndex) => wire(from, node, inputIndex));
    return node;
  };

  const backward = [
    backwardBlock('sum', {
      manualBackpropUpstream: 'gL',
      manualBackpropGradient0: 'gm'
    }, [matvec]),
    backwardBlock('matvec', {
      manualBackpropUpstream: 'gm',
      manualBackpropGradient0: 'gA',
      manualBackpropGradient1: 'gk'
    }, [unfold, k]),
    backwardBlock('unfold2d', {
      manualBackpropUpstream: 'gA',
      manualBackpropGradient0: 'gx'
    }, [x])
  ];

  // 둘 다 계산 pins the order: forward first, then each backward block.
  let body = loss;
  for (const node of backward) {
    const sequence = make('sequence');
    wire(body, sequence, 0);
    wire(node, sequence, 1);
    body = sequence;
  }

  writeRuntimeVariable('x', arrayValue(new Float32Array([
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  ]), [3, 3]), true);
  writeRuntimeVariable('k', arrayValue(new Float32Array([1, 0, 0, 1]), [4]), true);

  await evaluateGraph();
  const failed = [...graph.nodes.values()].find(node => node.lastError);
  if (failed) throw new Error(`${failed.type}: ${failed.lastError}`);

  const gradK = RUNTIME_VARIABLES.get('gk').value;
  const gradX = RUNTIME_VARIABLES.get('gx').value;
  assert(arraysEqual(Array.from(gradK.data), [12, 16, 24, 28]), '커널 gradient가 틀렸습니다.');
  assert(arraysEqual(Array.from(gradX.data), [
    1, 1, 0,
    1, 2, 1,
    0, 1, 1
  ]), '입력까지 전달되는 합성곱 gradient가 틀렸습니다.');
  assert(gradX.shape[0] === 3 && gradX.shape[1] === 3, '입력 gradient 모양이 틀렸습니다.');

  return {
    reshape: 'OK',
    unfold: 'OK',
    fold: 'OK',
    overlapGradient: 'OK',
    padding: 'OK',
    graphManualBackprop: 'OK'
  };
}

const chromium = loadChromium();
const { server, port } = await startServer();
let browser;

try {
  browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE_PATH || undefined });
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });

  if (pageErrors.length) throw new Error(`페이지 로딩 오류: ${pageErrors.join(' | ')}`);
  const result = await page.evaluate(checkInPage);
  console.log('spatial math checks:', result);
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
