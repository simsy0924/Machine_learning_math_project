#!/usr/bin/env node
// Correctness check for the result buffer arena.
//
// The arena recycles Float32Array buffers between training iterations. A bug
// there does not throw — it silently trains on stale numbers. So instead of
// asserting hand-derived expectations, every graph below is run twice, once
// with pooling disabled and once with it enabled, and the resulting runtime
// variable state must be bit-identical.
//
// The graphs deliberately stress the cases the pooling rule has to survive:
// blocks that hand an input buffer straight back (펼치기, 값 보기, 둘 다 계산),
// values that live across iterations (vector and scalar accumulators),
// user-defined blocks inside the loop, and nested repeats.

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function loadChromium() {
  const globalRoot = path.join(path.dirname(process.execPath), '..', 'lib', 'node_modules');
  for (const candidate of [process.env.PLAYWRIGHT_MODULE, 'playwright', path.join(globalRoot, 'playwright')].filter(Boolean)) {
    try {
      return require(candidate).chromium;
    } catch (error) {
      if (error.code !== 'MODULE_NOT_FOUND') throw error;
    }
  }
  throw new Error('playwright를 찾지 못했습니다.');
}

const CONTENT_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.bin': 'application/octet-stream'
};

function startServer() {
  const server = http.createServer((request, response) => {
    const requested = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    const filePath = path.resolve(ROOT, requested === '/' ? 'index.html' : requested.replace(/^\/+/, ''));
    if (filePath !== ROOT && !filePath.startsWith(ROOT + path.sep)) return response.writeHead(403).end();
    fs.readFile(filePath, (error, data) => {
      if (error) return response.writeHead(404).end();
      response.writeHead(200, { 'content-type': CONTENT_TYPES[path.extname(filePath)] || 'application/octet-stream' });
      response.end(data);
    });
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port })));
}

function runInPage(classes) {
  const build = {};

  // ---- shared helpers ----
  const add = (type, params) => {
    const node = addBlock(type);
    if (params) Object.assign(node.params, params);
    return node;
  };
  const link = (from, to, inputIndex = 0) => graph.connections.push({ from: from.id, to: to.id, inputIndex });

  function stateChecksum() {
    let hashA = 2166136261 >>> 0;
    let hashB = 2246822519 >>> 0;
    const mix = (word, index) => {
      hashA = Math.imul(hashA ^ word, 16777619) >>> 0;
      hashB = Math.imul(hashB ^ (word + index), 3266489917) >>> 0;
    };
    for (const name of Array.from(RUNTIME_VARIABLES.keys()).sort()) {
      for (let i = 0; i < name.length; i++) mix(name.charCodeAt(i), i);
      const value = RUNTIME_VARIABLES.get(name).value;
      if (typeof value === 'number') {
        const words = new Uint32Array(new Float64Array([value]).buffer);
        mix(words[0], 0);
        mix(words[1], 1);
      } else {
        const words = new Uint32Array(value.data.buffer, value.data.byteOffset, value.data.length);
        for (let i = 0; i < words.length; i++) mix(words[i], i);
      }
    }
    return hashA.toString(16).padStart(8, '0') + hashB.toString(16).padStart(8, '0');
  }

  // ---- graph 1: aliasing blocks feeding a persistent accumulator ----
  // 펼치기 shares its input's buffer, 값 보기 returns its input, and 둘 다 계산
  // returns its second input. All three sit between a pooled kernel result and
  // a 값 바꾸기, which is exactly the aliasing the pooling rule must survive.
  build.aliasingAccumulator = () => {
    resetWorkspace();
    const dataset = add('dataset');
    const index = add('variable', { name: 'i', mode: 'scalar', value: 0 });
    const sample = add('datasetSampleByIndex');
    link(dataset, sample, 0);
    link(index, sample, 1);
    const image = add('sampleImage');
    link(sample, image, 0);
    const x = add('flatten');
    link(image, x, 0);

    // acc ← 값 보기(펼치기(acc + x)) : a vector that survives every iteration
    const acc = add('variable', { name: 'acc', mode: 'vector', length: 784, init: 'constant', value: 0 });
    const sum = add('add');
    link(acc, sum, 0);
    link(x, sum, 1);
    const flattened = add('flatten');
    link(sum, flattened, 0);
    const shown = add('display');
    link(flattened, shown, 0);
    const setAcc = add('setVariable', { variable: 'acc' });
    link(shown, setAcc, 0);

    // total ← total + 합(acc + x)
    const total = add('variable', { name: 'total', mode: 'scalar', value: 0 });
    const rowSum = add('sum');
    link(shown, rowSum, 0);
    const newTotal = add('add');
    link(total, newTotal, 0);
    link(rowSum, newTotal, 1);
    const setTotal = add('setVariable', { variable: 'total' });
    link(newTotal, setTotal, 0);

    const both = add('sequence');
    link(setAcc, both, 0);
    link(setTotal, both, 1);
    const repeat = add('repeat', { count: 120, start: 0, indexVariable: 'i' });
    link(both, repeat, 0);
    return repeat;
  };

  // ---- graph 2: training with a user-defined block inside the loss ----
  build.userBlockTraining = () => {
    resetWorkspace();
    const dataset = add('dataset');
    const index = add('variable', { name: 'i', mode: 'scalar', value: 0 });
    const sample = add('datasetSampleByIndex');
    link(dataset, sample, 0);
    link(index, sample, 1);
    const image = add('sampleImage');
    link(sample, image, 0);
    const x = add('flatten');
    link(image, x, 0);

    const W = add('variable', {
      name: 'W', mode: 'matrix', rows: classes.length, cols: 784,
      init: 'random', seed: 3, scale: 0.05
    });
    const z = add('matvec');
    link(W, z, 0);
    link(x, z, 1);

    // softmax(z) grouped into a user block, so the loss differentiates through
    // evaluateUserDefinition / userBlockVJP rather than the compiled path.
    const zMax = add('arrayMax');
    link(z, zMax, 0);
    const shifted = add('subtract');
    link(z, shifted, 0);
    link(zMax, shifted, 1);
    const expZ = add('exp');
    link(shifted, expZ, 0);
    const expSum = add('sum');
    link(expZ, expSum, 0);
    const p = add('divide');
    link(expZ, p, 0);
    link(expSum, p, 1);

    startGroupSelection();
    for (const node of [zMax, shifted, expZ, expSum, p]) toggleGroupNode(node.id);
    createUserBlockFromSelection();
    const softmax = [...graph.nodes.values()].find(node => node.type.startsWith('custom:'));
    if (!softmax) throw new Error('사용자 블록을 만들지 못했습니다.');

    const logP = add('log');
    link(softmax, logP, 0);
    const y = add('sampleOneHot');
    link(sample, y, 0);
    const yLogP = add('multiply');
    link(y, yLogP, 0);
    link(logP, yLogP, 1);
    const crossEntropy = add('sum');
    link(yLogP, crossEntropy, 0);
    const minusOne = add('number', { value: -1 });
    const loss = add('multiply');
    link(crossEntropy, loss, 0);
    link(minusOne, loss, 1);

    const lr = add('number', { value: 0.05 });
    const gradient = add('derivative', { variable: 'W' });
    link(loss, gradient, 0);
    const scaled = add('multiply');
    link(lr, scaled, 0);
    link(gradient, scaled, 1);
    const updated = add('subtract');
    link(W, updated, 0);
    link(scaled, updated, 1);
    const assign = add('setVariable', { variable: 'W' });
    link(updated, assign, 0);

    const repeat = add('repeat', { count: 150, start: 0, indexVariable: 'i' });
    link(assign, repeat, 0);
    return repeat;
  };

  // ---- graph 3: a matrix accumulator that survives every iteration ----
  // Elementwise kernels happen to stay correct even if an output buffer aliases
  // an input, because they read and write the same index. 외적 does not: it
  // writes a whole row from one scalar. Feeding a retained matrix through it
  // makes this graph sensitive to a buffer being recycled while still live.
  build.matrixAccumulator = () => {
    resetWorkspace();
    const M = add('variable', { name: 'M', mode: 'matrix', rows: 16, cols: 16, init: 'constant', value: 0.5 });
    const v = add('variable', { name: 'v', mode: 'vector', length: 16, init: 'random', seed: 5 });
    const index = add('variable', { name: 'i', mode: 'scalar', value: 0 });

    const projected = add('matvec');
    link(M, projected, 0);
    link(v, projected, 1);
    const outer = add('outerProduct');
    link(projected, outer, 0);
    link(v, outer, 1);
    const grown = add('add');
    link(outer, grown, 0);
    link(M, grown, 1);
    const setM = add('setVariable', { variable: 'M' });
    link(grown, setM, 0);

    const total = add('variable', { name: 'total', mode: 'scalar', value: 0 });
    const rowSum = add('sum');
    link(projected, rowSum, 0);
    const scaledSum = add('multiply');
    link(rowSum, scaledSum, 0);
    link(index, scaledSum, 1);
    const newTotal = add('add');
    link(total, newTotal, 0);
    link(scaledSum, newTotal, 1);
    const setTotal = add('setVariable', { variable: 'total' });
    link(newTotal, setTotal, 0);

    const both = add('sequence');
    link(setM, both, 0);
    link(setTotal, both, 1);
    const repeat = add('repeat', { count: 30, start: 0, indexVariable: 'i' });
    link(both, repeat, 0);
    return repeat;
  };

  // ---- graph 4: nested repeats over a shared accumulator ----
  build.nestedRepeat = () => {
    resetWorkspace();
    const acc = add('variable', { name: 'acc', mode: 'vector', length: 16, init: 'constant', value: 1 });
    const inner = add('variable', { name: 'i', mode: 'scalar', value: 0 });
    const outer = add('variable', { name: 'epoch', mode: 'scalar', value: 0 });

    const scaled = add('multiply');
    link(acc, scaled, 0);
    link(inner, scaled, 1);
    const shifted = add('add');
    link(scaled, shifted, 0);
    link(outer, shifted, 1);
    const capped = add('maximum');
    const ceiling = add('number', { value: 5 });
    link(ceiling, capped, 0);
    link(shifted, capped, 1);
    const setAcc = add('setVariable', { variable: 'acc' });
    link(capped, setAcc, 0);

    const innerRepeat = add('repeat', { count: 7, start: 1, indexVariable: 'i' });
    link(setAcc, innerRepeat, 0);
    const outerRepeat = add('repeat', { count: 5, start: 0, indexVariable: 'epoch' });
    link(innerRepeat, outerRepeat, 0);
    return outerRepeat;
  };

  return (async () => {
    for (const box of document.querySelectorAll('#classPicker input[type=checkbox]')) {
      box.checked = classes.includes(box.value);
    }
    await window.quickDrawDataset.loadSelectedClasses();

    const originalWithResultArena = withResultArena;
    const report = [];

    for (const [name, make] of Object.entries(build)) {
      const checksums = [];

      for (const pooling of [false, true]) {
        // With pooling off every kernel allocates a fresh buffer, which is the
        // behaviour the arena has to reproduce exactly.
        withResultArena = pooling
          ? originalWithResultArena
          : (arena, run) => run();

        try {
          const repeat = make();
          RUNTIME_VARIABLES.clear();
          pendingVariableUpdates = null;
          sharedAutodiffContextKey = null;
          sharedAutodiffByOutput = new Map();

          selectNode(repeat.id);
          await evaluateGraph();

          const failed = [...graph.nodes.values()].find(node => node.lastError);
          if (failed) throw new Error(`${getBlockDef(failed.type).title}: ${failed.lastError}`);
          checksums.push(stateChecksum());
        } finally {
          withResultArena = originalWithResultArena;
        }
      }

      report.push({
        name,
        pooledOff: checksums[0],
        pooledOn: checksums[1],
        match: checksums[0] === checksums[1]
      });
    }

    return report;
  })();
}

async function main() {
  const classes = ['cat', 'fish', 'house'];
  const chromium = loadChromium();
  const { server, port } = await startServer();
  const browser = await chromium.launch();

  try {
    const page = await browser.newPage();
    const failures = [];
    page.on('pageerror', error => failures.push(String(error)));
    page.on('dialog', dialog => dialog.accept('softmax'));

    await page.goto(`http://127.0.0.1:${port}/index.html`, { waitUntil: 'load' });
    await page.waitForFunction(() => typeof window.quickDrawDataset !== 'undefined');

    const report = await page.evaluate(runInPage, classes);

    let failed = 0;
    for (const row of report) {
      if (!row.match) failed++;
      console.log(
        `${row.match ? 'OK  ' : '불일치'} ${row.name.padEnd(20)} ` +
        `풀링 끔 ${row.pooledOff} · 풀링 켬 ${row.pooledOn}`
      );
    }

    if (failures.length) {
      console.log('\n페이지 오류:');
      for (const failure of failures) console.log(`  ${failure}`);
    }

    if (failed) {
      console.error(`\n${failed}개 그래프에서 버퍼 재사용이 결과를 바꿨습니다.`);
      process.exitCode = 1;
    } else {
      console.log('\n모든 그래프에서 버퍼 재사용 전후 결과가 비트 단위로 같습니다.');
    }
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
