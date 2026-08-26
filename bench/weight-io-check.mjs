#!/usr/bin/env node
// Regression checks for model weight selection in direct-SGD and minibatch graphs.

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
      response.writeHead(200, { 'content-type': path.extname(filePath) === '.js' ? 'text/javascript' : 'text/html' });
      response.end(data);
    });
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port })));
}

function loadChromium() {
  const globalRoot = path.join(path.dirname(process.execPath), '..', 'lib', 'node_modules');
  for (const candidate of [process.env.PLAYWRIGHT_MODULE, 'playwright', path.join(globalRoot, 'playwright')].filter(Boolean)) {
    try { return require(candidate).chromium; }
    catch (error) { if (error.code !== 'MODULE_NOT_FOUND') throw error; }
  }
  throw new Error('playwright를 찾지 못했습니다.');
}

function checkInPage() {
  function assert(condition, message) {
    if (!condition) throw new Error(message);
  }

  function addVariable(name, mode = 'vector', length = 2) {
    const node = addBlock('variable');
    Object.assign(node.params, {
      name,
      mode,
      length,
      rows: 2,
      cols: 2,
      init: 'constant',
      value: 0
    });
    return node;
  }

  function addDerivative(variable) {
    const node = addBlock('derivative');
    node.params.variable = variable;
    return node;
  }

  function addSetter(variable) {
    const node = addBlock('setVariable');
    node.params.variable = variable;
    return node;
  }

  // Direct SGD: derivative target and setVariable target are the same parameter.
  resetWorkspace();
  addVariable('W');
  addDerivative('W');
  addSetter('W');
  assert(JSON.stringify(collectTrainableVariableNames()) === JSON.stringify(['W']), 'direct SGD에서 W를 학습 가중치로 찾지 못했습니다.');

  // Minibatch SGD: derivative targets W, gradient is accumulated into gW, and W
  // is updated later from gW. Only W is a model parameter.
  resetWorkspace();
  const w = addVariable('W');
  addVariable('gW');
  addDerivative('W');
  addSetter('gW');
  addSetter('W');

  const names = collectTrainableVariableNames();
  assert(names.length === 1 && names[0] === 'W', `미니배치 학습 가중치 판별 오류: ${names.join(', ')}`);

  const trained = fastArrayValue(new Float32Array([1.25, -2.5]), [2]);
  writeRuntimeVariable('W', trained, true);
  writeRuntimeVariable('gW', fastArrayValue(new Float32Array([0, 0]), [2]), true);

  const snapshot = buildWeightsSnapshot();
  assert(snapshot.variables.length === 1, '가중치 스냅샷에 accumulator가 포함되었습니다.');
  assert(snapshot.variables[0].name === 'W', `가중치 스냅샷이 ${snapshot.variables[0].name}을 저장했습니다.`);

  // A legacy broken file containing only gW must fail instead of silently
  // restoring the zero accumulator while leaving W at its initial value.
  const broken = {
    ...snapshot,
    variables: [{ name: 'gW', value: serializeRuntimeValue(fastArrayValue(new Float32Array([0, 0]), [2])) }]
  };
  let rejected = false;
  try { restoreWeights(broken); }
  catch (error) { rejected = /빠진 학습 가중치|학습 가중치가 아닌 변수/.test(error.message); }
  assert(rejected, 'accumulator-only 가중치 파일을 거부하지 않았습니다.');

  // A correct file restores W exactly.
  writeRuntimeVariable('W', fastArrayValue(new Float32Array([9, 9]), [2]), true);
  restoreWeights(snapshot);
  const restored = readRuntimeVariable(w);
  assert(restored.data[0] === 1.25 && restored.data[1] === -2.5, '실제 학습 가중치를 정확히 복원하지 못했습니다.');

  return {
    directSgd: 'OK',
    minibatchSelection: 'OK',
    accumulatorRejected: 'OK',
    restoreActualWeight: 'OK'
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
  if (pageErrors.length) throw new Error(pageErrors.join(' | '));
  const result = await page.evaluate(checkInPage);
  console.log('weight IO checks:', result);
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
