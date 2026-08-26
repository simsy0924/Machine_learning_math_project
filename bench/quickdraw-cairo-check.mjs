#!/usr/bin/env node
// Browser integration check for vector capture -> Quick Draw simplification -> Cairo WASM -> 28x28.

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
  '.bin': 'application/octet-stream',
  '.wasm': 'application/wasm'
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
      response.writeHead(200, {
        'content-type': CONTENT_TYPES[path.extname(filePath)] || 'application/octet-stream',
        'cache-control': 'no-store'
      });
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

const chromium = loadChromium();
const { server, port } = await startServer();
let browser;

try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));

  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });
  await page.evaluate(async () => {
    if (!window.QUICKDRAW_CAIRO) throw new Error('QUICKDRAW_CAIRO API가 없습니다.');
    await window.QUICKDRAW_CAIRO.ready;
  });

  const box = await page.locator('#drawCanvas').boundingBox();
  if (!box) throw new Error('drawCanvas 위치를 찾지 못했습니다.');

  // Draw a multi-segment shape through the same pointer path used by a person.
  const points = [
    [0.18, 0.76],
    [0.34, 0.24],
    [0.52, 0.68],
    [0.70, 0.28],
    [0.84, 0.74]
  ];
  const [firstX, firstY] = points[0];
  await page.mouse.move(box.x + box.width * firstX, box.y + box.height * firstY);
  await page.mouse.down();
  for (const [x, y] of points.slice(1)) {
    await page.mouse.move(box.x + box.width * x, box.y + box.height * y, { steps: 12 });
  }
  await page.mouse.up();

  const result = await page.evaluate(() => {
    const value = preprocessCanvasTo28(drawCanvas);
    const data = Array.from(value.data);
    const nonzero = data.reduce((count, pixel) => count + (pixel > 0 ? 1 : 0), 0);
    const sum = data.reduce((total, pixel) => total + pixel, 0);
    return {
      shape: Array.from(value.shape),
      length: data.length,
      nonzero,
      sum,
      min: Math.min(...data),
      max: Math.max(...data),
      version: window.QUICKDRAW_CAIRO.version,
      rawStrokeCount: window.QUICKDRAW_CAIRO.rawStrokeCount,
      runtimeState: document.documentElement.dataset.quickDrawCairo
    };
  });

  if (pageErrors.length) throw new Error(`페이지 오류: ${pageErrors.join(' | ')}`);
  if (result.runtimeState !== 'ready') throw new Error(`Cairo runtime 상태: ${result.runtimeState}`);
  if (result.version !== '1.18.4') throw new Error(`예상하지 못한 Cairo 버전: ${result.version}`);
  if (result.length !== 784 || result.shape[0] !== 28 || result.shape[1] !== 28) {
    throw new Error(`28×28 출력 모양이 아닙니다: ${JSON.stringify(result.shape)} / ${result.length}`);
  }
  if (result.rawStrokeCount !== 1) throw new Error(`벡터 stroke 기록 수가 틀렸습니다: ${result.rawStrokeCount}`);
  if (result.nonzero < 10 || result.sum <= 0 || result.max <= 0 || result.max > 1 || result.min < 0) {
    throw new Error(`Cairo 출력 픽셀 범위/내용이 이상합니다: ${JSON.stringify(result)}`);
  }

  console.log('quickdraw cairo browser check:', result);
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
