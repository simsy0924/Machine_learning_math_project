#!/usr/bin/env node
// Reproducible training-speed benchmark for the math block workspace.
//
// It serves the repository over localhost, drives the real page in headless
// Chromium, builds a fixed softmax-regression training graph out of ordinary
// blocks, and runs it. Two numbers come back:
//
//   - step/s   : how fast the training repeat runs
//   - checksum : a bit-exact hash of the trained weights
//
// The checksum is the safety net for optimization work. Any change that only
// removes overhead must leave it byte-for-byte identical.

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// ---------- options ----------

function parseArgs(argv) {
  const options = {
    steps: 10000,
    runs: 3,
    warmup: 1,
    hidden: 0,
    classes: ['cat', 'fish', 'house'],
    lr: 0.05,
    seed: 1,
    headed: false,
    json: false
  };

  for (let i = 0; i < argv.length; i++) {
    const [flag, inlineValue] = argv[i].split('=');
    const value = inlineValue ?? argv[++i];
    switch (flag) {
      case '--steps': options.steps = Number(value); break;
      case '--runs': options.runs = Number(value); break;
      case '--warmup': options.warmup = Number(value); break;
      case '--hidden': options.hidden = Number(value); break;
      case '--classes': options.classes = String(value).split(','); break;
      case '--lr': options.lr = Number(value); break;
      case '--seed': options.seed = Number(value); break;
      case '--headed': options.headed = true; i--; break;
      case '--json': options.json = true; i--; break;
      case '--help':
        console.log('usage: node bench/train-bench.mjs [--steps N] [--runs N] [--warmup N] [--hidden N] [--classes a,b,c] [--lr X] [--seed N] [--headed] [--json]');
        process.exit(0);
        break;
      default:
        throw new Error(`알 수 없는 옵션: ${flag}`);
    }
  }
  return options;
}

// ---------- static server ----------

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

    // Never serve anything outside the repository.
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

// ---------- playwright lookup ----------

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

// ---------- page side ----------

// Runs inside the page. Only uses the app's own public globals, so it exercises
// exactly the code path a user hits with the 계산 button.
function benchmarkInPage(options) {
  const { classes, steps, lr, seed, runs, warmup, hidden } = options;

  function fnv(hashA, hashB, word, index) {
    return [
      Math.imul(hashA ^ word, 16777619) >>> 0,
      Math.imul(hashB ^ (word + index), 3266489917) >>> 0
    ];
  }

  // Bit-exact hash of every runtime variable, so a pure-overhead optimization
  // is provably value-preserving.
  function weightChecksum() {
    let hashA = 2166136261 >>> 0;
    let hashB = 2246822519 >>> 0;

    for (const name of Array.from(RUNTIME_VARIABLES.keys()).sort()) {
      for (let i = 0; i < name.length; i++) {
        [hashA, hashB] = fnv(hashA, hashB, name.charCodeAt(i), i);
      }

      const value = RUNTIME_VARIABLES.get(name).value;
      if (typeof value === 'number') {
        const words = new Uint32Array(new Float64Array([value]).buffer);
        [hashA, hashB] = fnv(hashA, hashB, words[0], 0);
        [hashA, hashB] = fnv(hashA, hashB, words[1], 1);
      } else {
        const words = new Uint32Array(value.data.buffer, value.data.byteOffset, value.data.length);
        for (let i = 0; i < words.length; i++) [hashA, hashB] = fnv(hashA, hashB, words[i], i);
      }
    }

    return hashA.toString(16).padStart(8, '0') + hashB.toString(16).padStart(8, '0');
  }

  function buildTrainingGraph() {
    resetWorkspace();

    const add = (type, params) => {
      const node = addBlock(type);
      if (params) Object.assign(node.params, params);
      return node;
    };
    const link = (from, to, inputIndex = 0) => {
      graph.connections.push({ from: from.id, to: to.id, inputIndex });
    };

    const classCount = classes.length;

    // sample -> x
    const dataset = add('dataset');
    const index = add('variable', { name: 'i', mode: 'scalar', value: 0 });
    const sample = add('datasetSampleByIndex');
    link(dataset, sample, 0);
    link(index, sample, 1);
    const image = add('sampleImage');
    link(sample, image, 0);
    const x = add('flatten');
    link(image, x, 0);

    // One affine layer: name·input + bias. The parameters are collected so the
    // SGD update below can be generated for each of them.
    const parameters = [];
    const affine = (input, name, rows, cols, layerSeed) => {
      const weights = add('variable', {
        name: `W${name}`, mode: 'matrix', rows, cols,
        init: 'random', seed: layerSeed, scale: 0.05
      });
      const product = add('matvec');
      link(weights, product, 0);
      link(input, product, 1);

      const bias = add('variable', {
        name: `b${name}`, mode: 'vector', length: rows, init: 'constant', value: 0
      });
      const sum = add('add');
      link(product, sum, 0);
      link(bias, sum, 1);

      parameters.push(weights, bias);
      return sum;
    };

    // With --hidden N the graph becomes a two-layer network with a ReLU built
    // from the ordinary 최댓값 block, which is what the project is actually for.
    let features = x;
    let inputSize = 784;
    if (hidden > 0) {
      const preActivation = affine(x, '1', hidden, 784, seed);
      const zero = add('number', { value: 0 });
      const relu = add('maximum');
      link(zero, relu, 0);
      link(preActivation, relu, 1);
      features = relu;
      inputSize = hidden;
    }

    const z = affine(features, hidden > 0 ? '2' : '', classCount, inputSize, hidden > 0 ? seed + 1 : seed);

    // p = softmax(z), built from ordinary blocks and stabilized by max(z)
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

    // loss = -Σ y·ln(p)
    const logP = add('log');
    link(p, logP, 0);
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

    // SGD update for both parameters, committed together at the end of a step
    const learningRate = add('number', { value: lr });
    const updates = [];
    for (const parameter of parameters) {
      const name = parameter.params.name;
      const gradient = add('derivative', { variable: name });
      link(loss, gradient, 0);
      const scaled = add('multiply');
      link(learningRate, scaled, 0);
      link(gradient, scaled, 1);
      const updated = add('subtract');
      link(parameter, updated, 0);
      link(scaled, updated, 1);
      const assign = add('setVariable', { variable: name });
      link(updated, assign, 0);
      updates.push(assign);
    }

    // 둘 다 계산 takes two inputs, so chain it to commit every parameter in one
    // iteration.
    let body = updates[0];
    for (let i = 1; i < updates.length; i++) {
      const sequence = add('sequence');
      link(body, sequence, 0);
      link(updates[i], sequence, 1);
      body = sequence;
    }

    const repeat = add('repeat', { count: steps, start: 0, indexVariable: 'i' });
    link(body, repeat, 0);

    return { loss, repeat, index };
  }

  // evaluateGraph walks nodes in insertion order, so the loss block is computed
  // before the repeat block that trains the weights. Probe the loss separately
  // afterwards, on samples the training range never touched, to confirm the
  // benchmark graph actually learns something.
  function heldOutLoss(lossId, indexNode, sampleCount = 300) {
    const perClass = Math.floor(steps / classes.length);
    if (perClass + sampleCount >= 10000) return null;

    const signature = variableSignature(indexNode);
    const first = (perClass + 500) * classes.length;
    let total = 0;

    for (let i = 0; i < sampleCount; i++) {
      setRuntimeVariableEntry('i', { value: first + i, signature });
      total += evaluateNode(lossId, new Map(), new Set());
    }
    return total / sampleCount;
  }

  // Every run must start from the same state or the checksums diverge.
  function resetRuntimeState() {
    clearRuntimeVariables();
    pendingVariableUpdates = null;
    sharedAutodiffContextKey = null;
    sharedAutodiffByOutput = new Map();
  }

  return (async () => {
    const picker = document.querySelectorAll('#classPicker input[type="checkbox"]');
    for (const box of picker) box.checked = classes.includes(box.value);
    await window.quickDrawDataset.loadSelectedClasses();

    const loaded = window.quickDrawDataset.getLoadedClassNames();
    if (loaded.length !== classes.length) {
      throw new Error(`데이터 로드 실패: ${loaded.join(',') || '없음'}`);
    }

    const { loss, index } = buildTrainingGraph();
    const results = [];

    // The first runs are discarded: the JIT has not warmed up on the compiled
    // execution plans yet and they read 30-40% slow.
    for (let run = 0; run < warmup + runs; run++) {
      resetRuntimeState();
      const started = performance.now();
      await evaluateGraph();
      const elapsedMs = performance.now() - started;

      const lossNode = graph.nodes.get(loss.id);
      if (lossNode?.lastError) throw new Error(`그래프 오류: ${lossNode.lastError}`);
      if (run < warmup) continue;

      const checksum = weightChecksum();
      results.push({
        elapsedMs,
        stepsPerSecond: steps / (elapsedMs / 1000),
        msPerStep: elapsedMs / steps,
        checksum,
        heldOutLoss: heldOutLoss(loss.id, index)
      });
    }

    return { classes: loaded, steps, warmup, hidden, results };
  })();
}

// ---------- driver ----------

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const chromium = loadChromium();
  const { server, port } = await startServer();
  const browser = await chromium.launch({ headless: !options.headed });

  try {
    const page = await browser.newPage();
    const failures = [];
    page.on('pageerror', error => failures.push(String(error)));
    page.on('console', message => {
      if (message.type() === 'error') failures.push(message.text());
    });

    await page.goto(`http://127.0.0.1:${port}/index.html`, { waitUntil: 'load' });
    await page.waitForFunction(() => typeof window.quickDrawDataset !== 'undefined');

    const report = await page.evaluate(benchmarkInPage, {
      classes: options.classes,
      steps: options.steps,
      lr: options.lr,
      seed: options.seed,
      runs: options.runs,
      warmup: options.warmup,
      hidden: options.hidden
    });

    if (options.json) {
      console.log(JSON.stringify({ ...report, failures }, null, 2));
    } else {
      printReport(report, options, failures);
    }

    const checksums = new Set(report.results.map(result => result.checksum));
    if (checksums.size !== 1) {
      console.error('\n경고: 실행마다 체크섬이 다릅니다. 학습이 결정적이지 않습니다.');
      process.exitCode = 1;
    }
  } finally {
    await browser.close();
    server.close();
  }
}

function printReport(report, options, failures) {
  const rates = report.results.map(result => result.stepsPerSecond);
  const best = Math.max(...rates);
  const median = [...rates].sort((a, b) => a - b)[Math.floor(rates.length / 2)];

  console.log(
    `종류 ${report.classes.join(', ')} · step ${report.steps.toLocaleString()}` +
    ` · ${report.hidden > 0 ? `은닉층 ${report.hidden}` : '은닉층 없음'}` +
    ` · lr ${options.lr} · seed ${options.seed} · warmup ${report.warmup}회 제외`
  );
  report.results.forEach((result, index) => {
    console.log(
      `  run ${index + 1}: ${Math.round(result.stepsPerSecond).toLocaleString()} step/s` +
      ` · ${result.msPerStep.toFixed(4)} ms/step`
    );
  });
  console.log(`  중앙값 ${Math.round(median).toLocaleString()} step/s · 최고 ${Math.round(best).toLocaleString()} step/s`);
  console.log(`  가중치 체크섬 ${report.results[0].checksum}`);

  const loss = report.results[0].heldOutLoss;
  if (loss != null) console.log(`  학습 후 held-out 손실 ${loss.toFixed(6)}`);

  if (failures.length) {
    console.log('\n페이지 오류:');
    for (const failure of failures) console.log(`  ${failure}`);
  }
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
