#!/usr/bin/env node
// Compare the training-time unfold -> matvec fusion with the established path.
// The graph deliberately matches the CNN presentation model: one shared 3x3
// unfold feeding four filters, followed by four 2x2/stride-2 average pools.

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

function checkInPage() {
  function assert(condition, message) {
    if (!condition) throw new Error(message);
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
    RUNTIME_VARIABLES.clear();

    const x = add('variable', { name: 'x', mode: 'matrix', rows: 28, cols: 28, init: 'constant', value: 0 });
    const patches = add('unfold2d', { kernelRows: 3, kernelCols: 3, strideRows: 1, strideCols: 1, padding: 0 });
    link(x, patches);

    const zero = add('number', { value: 0 });
    const avg = add('constantVector', { length: 4, value: 0.25 });
    const branchLosses = [];

    for (let filter = 0; filter < 4; filter++) {
      const k = add('variable', {
        name: `k${filter + 1}`,
        mode: 'vector',
        length: 9,
        init: 'constant',
        value: 0
      });
      const conv = add('matvec');
      link(patches, conv, 0);
      link(k, conv, 1);

      const shape = add('reshape', { shape: '26,26' });
      link(conv, shape);
      const relu = add('maximum');
      link(zero, relu, 0);
      link(shape, relu, 1);

      const poolWindows = add('unfold2d', { kernelRows: 2, kernelCols: 2, strideRows: 2, strideCols: 2, padding: 0 });
      link(relu, poolWindows);
      const pooled = add('matvec');
      link(poolWindows, pooled, 0);
      link(avg, pooled, 1);
      const branchLoss = add('sum');
      link(pooled, branchLoss);
      branchLosses.push(branchLoss);
    }

    let loss = branchLosses[0];
    for (let i = 1; i < branchLosses.length; i++) {
      const next = add('add');
      link(loss, next, 0);
      link(branchLosses[i], next, 1);
      loss = next;
    }

    const xData = new Float32Array(28 * 28);
    for (let i = 0; i < xData.length; i++) xData[i] = 0.2 + ((i * 37) % 101) / 125;
    writeRuntimeVariable('x', arrayValue(xData, [28, 28]), true);

    for (let filter = 0; filter < 4; filter++) {
      const data = new Float32Array(9);
      for (let i = 0; i < 9; i++) data[i] = (filter + 1) * 0.03 + (i + 1) * 0.01;
      writeRuntimeVariable(`k${filter + 1}`, arrayValue(data, [9]), true);
    }
    return loss;
  }

  function cloneGradients(map) {
    const copy = new Map();
    for (const [name, value] of map) copy.set(name, copyValue(value));
    return copy;
  }

  function sameFloat32Bits(a, b) {
    const aa = asArrayValue(a);
    const bb = asArrayValue(b);
    if (aa.data.length !== bb.data.length || aa.shape.length !== bb.shape.length) return false;
    for (let i = 0; i < aa.shape.length; i++) if (aa.shape[i] !== bb.shape[i]) return false;
    const aw = new Uint32Array(aa.data.buffer, aa.data.byteOffset, aa.data.length);
    const bw = new Uint32Array(bb.data.buffer, bb.data.byteOffset, bb.data.length);
    for (let i = 0; i < aw.length; i++) if (aw[i] !== bw[i]) return false;
    return true;
  }

  function benchmark(lossId, disabled, iterations = 120) {
    window.SPATIAL_FUSION_DISABLED = disabled;
    for (let i = 0; i < 12; i++) computeAllGraphGradients(lossId);
    const start = performance.now();
    for (let i = 0; i < iterations; i++) computeAllGraphGradients(lossId);
    return (performance.now() - start) / iterations;
  }

  assert(window.SPATIAL_TRAINING_FUSION, 'spatial training fusion runtime이 로드되지 않았습니다.');
  const loss = buildGraph();

  window.SPATIAL_FUSION_DISABLED = true;
  const baseline = cloneGradients(computeAllGraphGradients(loss.id));

  window.SPATIAL_FUSION_DISABLED = false;
  const fused = cloneGradients(computeAllGraphGradients(loss.id));
  assert(window.SPATIAL_TRAINING_FUSION.lastFusionCount === 8, `예상 fusion 8개, 실제 ${window.SPATIAL_TRAINING_FUSION.lastFusionCount}개`);

  assert(baseline.size === fused.size, 'gradient 변수 개수가 달라졌습니다.');
  for (const [name, expected] of baseline) {
    const actual = fused.get(name);
    assert(actual, `${name} gradient가 없습니다.`);
    assert(sameFloat32Bits(expected, actual), `${name} gradient 비트가 fusion 전후로 달라졌습니다.`);
  }

  const normalMs = benchmark(loss.id, true);
  const fusedMs = benchmark(loss.id, false);
  window.SPATIAL_FUSION_DISABLED = false;

  return {
    gradients: 'bit-exact',
    fusedMatvecs: window.SPATIAL_TRAINING_FUSION.lastFusionCount,
    normalMsPerPass: normalMs,
    fusedMsPerPass: fusedMs,
    speedup: normalMs / fusedMs
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
  console.log('spatial fusion checks:', result);
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
