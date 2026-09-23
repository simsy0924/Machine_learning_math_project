#!/usr/bin/env node
// node bench/presentation-content-check.mjs
//
// presentation/build.mjs 가 만드는 발표 단계 파일을 실제 페이지(헤드리스 크로뮴)에서
// 하나씩 불러와 '계산'을 실행하고, 모든 블록이 오류 없이 계산되는지와 값 보기 블록이
// 생성기가 예측한 값을 내는지 검사합니다. 디스크의 파일이 생성기와 다르면(스크립트를
// 고치고 다시 실행하지 않았으면) 그것도 실패로 봅니다.
//
// bench/presentation-check.mjs 는 발표 모드 전환 로직을, 이 파일은 발표 내용을 검사합니다.

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { buildAll } from '../presentation/build.mjs';

const require = createRequire(import.meta.url);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PRESENTATION = path.join(ROOT, 'presentation');

const CONTENT_TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.mmlab': 'application/json', '.bin': 'application/octet-stream', '.wasm': 'application/wasm'
};

function startServer() {
  const server = http.createServer((request, response) => {
    const requested = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    const relative = requested === '/' ? 'index.html' : requested.replace(/^\/+/, '');
    // Full Chromium asks for a favicon the app does not ship; that 404 is not a page error.
    if (requested === '/favicon.ico') { response.writeHead(204).end(); return; }
    const filePath = path.resolve(ROOT, relative);
    if (filePath !== ROOT && !filePath.startsWith(ROOT + path.sep)) { response.writeHead(403).end('forbidden'); return; }
    fs.readFile(filePath, (error, data) => {
      if (error) { response.writeHead(404).end('not found'); return; }
      response.writeHead(200, { 'content-type': CONTENT_TYPES[path.extname(filePath)] || 'application/octet-stream' });
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
  throw new Error('playwright를 찾지 못했습니다. `npm i -g playwright` 또는 PLAYWRIGHT_MODULE 환경변수로 경로를 지정하세요.');
}

// ---------- page side ----------

async function evaluateStepInPage({ project }) {
  await restoreProject(project, { loadDataset: false });
  await evaluateGraph();
  const serialize = value => {
    if (typeof value === 'number') return value;
    if (value && value.data && value.shape) return { shape: Array.from(value.shape), data: Array.from(value.data) };
    return value === undefined ? undefined : String(value);
  };
  return {
    status: document.getElementById('workspaceStatus').textContent,
    nodes: Array.from(graph.nodes.values()).map(node => ({ id: node.id, type: node.type, error: node.lastError || null, value: serialize(node.lastValue) }))
  };
}

// ---------- comparison ----------

function numbersOf(value) {
  if (typeof value === 'number') return [value];
  if (value && Array.isArray(value.data)) return value.data;
  return null;
}

function compare(actual, check) {
  const expected = check.expect !== undefined ? check.expect : check.approx;
  const tolerance = check.tol ?? 1e-6;
  const want = Array.isArray(expected) ? expected : [expected];
  const got = numbersOf(actual);
  if (!got) return `값이 없습니다: ${JSON.stringify(actual)}`;
  if (check.shape && JSON.stringify(actual.shape) !== JSON.stringify(check.shape)) return `모양 ${JSON.stringify(actual.shape)} ≠ ${JSON.stringify(check.shape)}`;
  if (!Array.isArray(expected) && typeof actual !== 'number') return `숫자를 기대했지만 배열입니다: ${JSON.stringify(actual)}`;
  if (got.length !== want.length) return `길이 ${got.length} ≠ ${want.length}`;
  for (let i = 0; i < want.length; i++) {
    if (!(Math.abs(got[i] - want[i]) <= tolerance)) return `[${i}] ${got[i]} ≠ ${want[i]} (허용 ${tolerance})`;
  }
  return null;
}

// ---------- driver ----------

async function main() {
  const all = buildAll();
  const failures = [];

  // 디스크 파일이 생성기와 같은지
  const manifest = JSON.parse(fs.readFileSync(path.join(PRESENTATION, 'steps.json'), 'utf8'));
  const manifestIds = manifest.steps.map(step => `${step.id}|${step.title}|${step.file}`).join('\n');
  const builtIds = all.steps.map(step => `${step.id}|${step.title}|${step.file}`).join('\n');
  if (manifestIds !== builtIds) failures.push('steps.json이 build.mjs와 다릅니다. node presentation/build.mjs 를 다시 실행하세요.');
  const descriptions = JSON.parse(fs.readFileSync(path.join(PRESENTATION, 'descriptions.json'), 'utf8'));
  if (JSON.stringify(descriptions) !== JSON.stringify(all.descriptions)) failures.push('descriptions.json이 build.mjs와 다릅니다.');
  for (const step of all.steps) {
    const onDisk = fs.readFileSync(path.join(PRESENTATION, step.file), 'utf8');
    if (onDisk !== JSON.stringify(step.project)) failures.push(`${step.file}이 build.mjs와 다릅니다.`);
  }

  const chromium = loadChromium();
  const { server, port } = await startServer();
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE_PATH || undefined });
  try {
    const page = await browser.newPage();
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') pageErrors.push(message.text()); });
    await page.goto(`http://127.0.0.1:${port}/index.html`, { waitUntil: 'load' });
    await page.waitForFunction(() => typeof window.quickDrawDataset !== 'undefined');

    for (const step of all.steps) {
      const started = Date.now();
      const result = await page.evaluate(evaluateStepInPage, { project: step.project });
      const elapsed = Date.now() - started;
      const problems = [];
      for (const node of result.nodes) if (node.error) problems.push(`노드 ${node.id} (${node.type}): ${node.error}`);
      const byId = new Map(result.nodes.map(node => [node.id, node]));
      for (const check of step.checks) {
        const node = byId.get(check.id);
        const problem = node ? compare(node.value, check) : '노드가 없습니다';
        if (problem) problems.push(`값 보기 ${check.id}: ${problem}`);
      }
      const label = `${step.file.padEnd(28)} 노드 ${String(result.nodes.length).padStart(3)} · 검사 ${String(step.checks.length).padStart(2)} · ${String(elapsed).padStart(5)} ms · ${result.status}`;
      if (problems.length) {
        console.log(`FAIL ${label}`);
        for (const problem of problems) console.log(`       ${problem}`);
        failures.push(`${step.file}: ${problems.length}개 문제`);
      } else {
        console.log(`OK   ${label}`);
      }
    }
    if (pageErrors.length) {
      failures.push(...pageErrors.map(text => `페이지 오류: ${text}`));
    }
  } finally {
    await browser.close();
    server.close();
  }

  if (failures.length) {
    console.log('\n실패:');
    for (const failure of failures) console.log(`  ${failure}`);
    process.exit(1);
  }
  console.log(`\n모든 단계 통과 (${all.steps.length}개)`);
}

main().catch(error => { console.error(error); process.exit(1); });
