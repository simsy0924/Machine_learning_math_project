#!/usr/bin/env node
// Bit-exact checks for the optimized dense/ReLU backward kernels.

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

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
      if (error) return response.writeHead(404).end('not found');
      response.writeHead(200).end(data);
    });
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port })));
}

function loadChromium() {
  const globalRoot = path.join(path.dirname(process.execPath), '..', 'lib', 'node_modules');
  for (const candidate of [process.env.PLAYWRIGHT_MODULE, 'playwright', path.join(globalRoot, 'playwright')].filter(Boolean)) {
    try { return require(candidate).chromium; } catch (error) { if (error.code !== 'MODULE_NOT_FOUND') throw error; }
  }
  throw new Error('playwright를 찾지 못했습니다.');
}

function checkInPage() {
  function sameFloat32(a, b) {
    if (!a || !b || a.length !== b.length) return false;
    const ua = new Uint32Array(a.buffer, a.byteOffset, a.length);
    const ub = new Uint32Array(b.buffer, b.byteOffset, b.length);
    for (let i = 0; i < ua.length; i++) if (ua[i] !== ub[i]) return false;
    return true;
  }

  function makeData(length, seed) {
    let state = seed >>> 0;
    const out = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      state = (1664525 * state + 1013904223) >>> 0;
      out[i] = (state / 4294967296) * 2 - 1;
    }
    return out;
  }

  function run(disabled, fn) {
    BACKWARD_FAST.setDisabled(disabled);
    return fn();
  }

  const matrix = arrayValue(makeData(64 * 169, 11), [64, 169]);
  const up64 = arrayValue(makeData(64, 12), [64]);
  const vec169 = arrayValue(makeData(169, 13), [169]);

  const transposeOld = run(true, () => transposeMatrixVectorValues(matrix, up64));
  const transposeFast = run(false, () => transposeMatrixVectorValues(matrix, up64));
  if (!sameFloat32(transposeOld.data, transposeFast.data)) throw new Error('전치 행렬 × 벡터 결과가 비트 단위로 다릅니다.');

  const outerOld = run(true, () => outerProductValues(up64, vec169));
  const outerFast = run(false, () => outerProductValues(up64, vec169));
  if (!sameFloat32(outerOld.data, outerFast.data)) throw new Error('외적 결과가 비트 단위로 다릅니다.');

  const reluInput = arrayValue(makeData(676, 14), [676]);
  reluInput.data[0] = 0;
  reluInput.data[17] = 0;
  const reluUp = arrayValue(makeData(676, 15), [676]);

  const reluOld = run(true, () => primitiveVJP('maximum', [0, reluInput], null, reluUp));
  const reluFast = run(false, () => primitiveVJP('maximum', [0, reluInput], null, reluUp));
  if (Object.is(reluOld[0], reluFast[0]) === false) throw new Error('ReLU scalar gradient가 다릅니다.');
  if (!sameFloat32(reluOld[1].data, reluFast[1].data)) throw new Error('ReLU array gradient가 비트 단위로 다릅니다.');

  const iterations = 1500;
  const measure = disabled => {
    BACKWARD_FAST.setDisabled(disabled);
    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
      transposeMatrixVectorValues(matrix, up64);
      outerProductValues(up64, vec169);
      primitiveVJP('maximum', [0, reluInput], null, reluUp);
    }
    return performance.now() - start;
  };

  // Warm both implementations before measuring.
  measure(true);
  measure(false);
  const oldMs = measure(true);
  const fastMs = measure(false);
  BACKWARD_FAST.setDisabled(false);

  return {
    bitExact: true,
    oldMs,
    fastMs,
    speedup: oldMs / fastMs
  };
}

const chromium = loadChromium();
const { server, port } = await startServer();
let browser;
try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });
  if (errors.length) throw new Error(errors.join(' | '));
  const result = await page.evaluate(checkInPage);
  console.log('backward fast checks:', result);
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
