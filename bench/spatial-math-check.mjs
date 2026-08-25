#!/usr/bin/env node
// Correctness checks for reshape + sliding-window unfold using the real browser app.

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

function checkInPage() {
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

  const ones = arrayValue(new Float32Array(16).fill(1), [4, 4]);
  const inputGradient = primitiveVJP('unfold2d', [image], unfolded, ones)[0];
  assert(arraysEqual(Array.from(inputGradient.data), [
    1, 2, 1,
    2, 4, 2,
    1, 2, 1
  ]), '겹치는 창의 unfold gradient 누적이 틀렸습니다.');

  unfoldNode.params = { kernelRows: 3, kernelCols: 3, strideRows: 1, strideCols: 1, padding: 1 };
  const padded = BLOCKS.unfold2d.compute(unfoldNode, [image]);
  assert(padded.shape[0] === 9 && padded.shape[1] === 9, 'padding unfold 출력 모양이 틀렸습니다.');
  assert(arraysEqual(Array.from(padded.data.slice(0, 9)), [0, 0, 0, 0, 1, 2, 0, 4, 5]), 'padding 값이 틀렸습니다.');

  // Full graph autodiff check: sum(unfold(x) @ k).
  resetWorkspace();
  const x = addBlock('variable');
  Object.assign(x.params, { name: 'x', mode: 'matrix', rows: 3, cols: 3, init: 'constant', value: 0 });
  const k = addBlock('variable');
  Object.assign(k.params, { name: 'k', mode: 'vector', length: 4, init: 'constant', value: 0 });
  const unfold = addBlock('unfold2d');
  Object.assign(unfold.params, { kernelRows: 2, kernelCols: 2, strideRows: 1, strideCols: 1, padding: 0 });
  const matvec = addBlock('matvec');
  const loss = addBlock('sum');
  graph.connections.push(
    { from: x.id, to: unfold.id, inputIndex: 0 },
    { from: unfold.id, to: matvec.id, inputIndex: 0 },
    { from: k.id, to: matvec.id, inputIndex: 1 },
    { from: matvec.id, to: loss.id, inputIndex: 0 }
  );

  writeRuntimeVariable('x', arrayValue(new Float32Array([
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  ]), [3, 3]), true);
  writeRuntimeVariable('k', arrayValue(new Float32Array([1, 0, 0, 1]), [4]), true);

  const gradK = differentiateGraph(loss.id, 'k');
  const gradX = differentiateGraph(loss.id, 'x');
  assert(arraysEqual(Array.from(gradK.data), [12, 16, 24, 28]), '커널 gradient가 틀렸습니다.');
  assert(arraysEqual(Array.from(gradX.data), [
    1, 1, 0,
    1, 2, 1,
    0, 1, 1
  ]), '입력까지 전달되는 합성곱 gradient가 틀렸습니다.');

  return {
    reshape: 'OK',
    unfold: 'OK',
    overlapGradient: 'OK',
    padding: 'OK',
    graphAutodiff: 'OK'
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
  console.log('spatial math checks:', result);
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
