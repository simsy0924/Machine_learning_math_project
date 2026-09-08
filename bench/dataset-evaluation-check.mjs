#!/usr/bin/env node
// Dataset-review regression checks: predictions, read-only state, filters, and cancellation.

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

async function checkInPage() {
  function assert(ok, message) { if (!ok) throw new Error(message); }
  graph.nodes.clear(); graph.connections = []; RUNTIME_VARIABLES.clear(); datasetCache.clear();
  for (const [name, levels] of [['cat', [204, 51, 128, 230]], ['fish', [26, 204, 128, 179]]]) {
    const bytes = new Uint8Array(4 * 784);
    levels.forEach((level, i) => { bytes[i * 784] = level; });
    datasetCache.set(name, { name, bytes, imageCount: 4 });
  }
  const image = addBlock('sampleImage'), flat = addBlock('flatten'), weight = addBlock('variable');
  Object.assign(weight.params, { name: 'reviewW', mode: 'matrix', rows: 2, cols: 784, init: 'constant', value: 0 });
  const w = new Float32Array(1568);w[0] = 1;w[784] = -1;
  writeRuntimeVariable('reviewW', arrayValue(w, [2, 784]), true);
  const mv = addBlock('matvec'), bias = addBlock('variable');
  Object.assign(bias.params, { name: 'reviewB', mode: 'vector', length: 2, init: 'constant', value: 0 });
  writeRuntimeVariable('reviewB', arrayValue(new Float32Array([0, 1]), [2]), true);
  const output = addBlock('add');
  for (const [from, to, inputIndex] of [[image.id,flat.id,0],[weight.id,mv.id,0],[flat.id,mv.id,1],[mv.id,output.id,0],[bias.id,output.id,1]]) graph.connections.push({from,to,inputIndex});
  selectedNodeId = output.id;
  // The source has deliberately no dataset edge. Review must cut it, not execute it.
  const before = JSON.stringify(Array.from(RUNTIME_VARIABLES));
  const classifier = createDatasetReviewPredictor(output.id);
  const pixel = arrayValue(new Float32Array(784).fill(0.25), [28,28]);
  const expected = classifier(pixel);
  writeRuntimeVariable('reviewW', arrayValue(new Float32Array(1568), [2,784]), true);
  assert(classifier(pixel).data[0] === expected.data[0], 'weights were not snapshotted');
  writeRuntimeVariable('reviewW', arrayValue(w, [2,784]), true);
  reviewElement('reviewStart').value = '0'; reviewElement('reviewCount').value = '8';
  await startDatasetReview();
  assert(datasetReviewRun.rows.length === 8, reviewElement('reviewSummary').textContent);
  assert(datasetReviewRun.rows[0].globalIndex === 0 && datasetReviewRun.rows[1].classIndex === 1 && datasetReviewRun.rows[2].index === 1, 'global index mapping');
  assert(datasetReviewRun.correct === 4 && datasetReviewRun.wrong === 4, 'wrong predictions/counts');
  assert(datasetReviewRun.credit === 4, 'accuracy numerator');
  assert(reviewElement('reviewSummary').textContent.includes('50.00%'), 'summary accuracy');
  assert(JSON.stringify(Array.from(RUNTIME_VARIABLES)) === before, 'review changed runtime variables');
  assert(reviewElement('reviewImage').querySelector('canvas'), 'missing image');
  reviewElement('reviewWrongRandom').click();
  assert(filteredDatasetReviews().every(row => row.credit === 0), 'random wrong contains correct image');
  assert(reviewElement('reviewImage').textContent.includes('오답'), 'wrong verdict missing');
  reviewElement('reviewClass').value = '0'; reviewElement('reviewClass').dispatchEvent(new Event('change'));
  assert(filteredDatasetReviews().every(row => row.classIndex === 0), 'class filter');
  reviewElement('reviewFilter').value = 'tie';reviewElement('reviewFilter').dispatchEvent(new Event('change'));
  assert(reviewElement('reviewImage').textContent.includes('해당하는 그림이 없습니다'), 'empty state');
  assert(reviewElement('reviewNext').disabled, 'empty navigation enabled');
  const tie = datasetReviewOutcome(arrayValue(new Float32Array([0.5,0.5]), [2]), 2, 1);
  assert(tie.credit === 0.5 && tie.status === 'tie', 'fractional tie credit');
  const noTruthTie = datasetReviewOutcome(arrayValue(new Float32Array([0.5,0.5,0]), [3]), 3, 2);
  assert(noTruthTie.credit === 0, 'tie excluding truth');
  let rejected = false;
  try { datasetReviewOutcome(arrayValue(new Float32Array([NaN,0]), [2]), 2, 0); } catch { rejected = true; }
  assert(rejected, 'NaN was accepted');
  rejected = false;
  try { datasetReviewOutcome(arrayValue(new Float32Array([1]), [1]), 2, 0); } catch { rejected = true; }
  assert(rejected, 'wrong output length accepted');
  output.params.manualBackpropMode = 'backward';rejected = false;
  try { createDatasetReviewPredictor(output.id); } catch { rejected = true; }
  assert(rejected, 'backward output accepted');delete output.params.manualBackpropMode;
  const setter = addBlock('setVariable');setter.params.variable = 'reviewB';
  graph.connections.push({ from: output.id, to: setter.id, inputIndex: 0 });rejected = false;
  try { createDatasetReviewPredictor(setter.id); } catch { rejected = true; }
  assert(rejected, 'stateful output accepted');
  selectedNodeId = output.id;
  reviewElement('reviewCount').value = '9';await startDatasetReview();
  assert(reviewElement('reviewSummary').textContent.includes('오류'), 'out-of-range wrapped');
  assert(datasetReviewRun.rows.length === 8, 'invalid run discarded previous result');
  // Force a UI yield after each image so a stop click deterministically interrupts.
  reviewElement('reviewCount').value = '8';
  const originalNow = performance.now; let clock = 0;
  performance.now = () => (clock += 60);
  setTimeout(() => reviewElement('reviewStop').click(), 0);
  await startDatasetReview();performance.now = originalNow;
  assert(datasetReviewRun.rows.length > 0 && datasetReviewRun.rows.length < 8, 'stop did not preserve partial results');
  assert(reviewElement('reviewSummary').textContent.includes('중단'), 'partial run marked complete');
  assert(!evaluateBtn.disabled && !evaluateSelectedBtn.disabled && reviewElement('reviewStop').disabled, 'controls not restored');
  return { accuracy: '50%', mapping: 'OK', weightSnapshot: 'OK', runtimeUnchanged: 'OK', wrongRandom: 'OK', filters: 'OK', ties: 'OK', invalidOutputs: 'OK', statefulRejected: 'OK', cancellation: 'OK' };
}

if (process.env.JSDOM_MODULE) {
  // Optional DOM-only fallback when a Chromium binary cannot be downloaded.
  // Canvas is stubbed; this checks events and results, not visual layout.
  const { JSDOM } = require(process.env.JSDOM_MODULE);
  const vm = await import('node:vm');
  const dom = new JSDOM(fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8'), {
    url: 'http://localhost/', runScripts: 'outside-only', pretendToBeVisual: true
  });
  const window = dom.window;
  window.HTMLCanvasElement.prototype.getContext = function () {
    return new Proxy({ createImageData: (width, height) => ({ data: new window.Uint8ClampedArray(width * height * 4) }) }, {
      get: (object, key) => key in object ? object[key] : () => {}
    });
  };
  window.alert = () => {}; window.confirm = () => true;
  window.fetch = async () => { throw new Error('Network omitted in DOM-only checks'); };
  try {
    const context = dom.getInternalVMContext();
    for (const script of window.document.querySelectorAll('script[src]')) {
      const file = script.getAttribute('src').split('?')[0];
      vm.runInContext(fs.readFileSync(path.join(ROOT, file), 'utf8'), context, { filename: file });
    }
    console.log('dataset review (DOM only):', await vm.runInContext(`(${checkInPage.toString()})()`, context));
  } finally { window.close(); }
} else {
const chromium = loadChromium();
const { server, port } = await startServer();
let browser;
try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });
  const result = await page.evaluate(checkInPage);
  if (errors.length) throw new Error(errors.join(' | '));
  console.log('dataset review:', result);
  if (process.env.REVIEW_SCREENSHOT) await page.locator('.dataset-review').screenshot({ path: process.env.REVIEW_SCREENSHOT });
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
}
