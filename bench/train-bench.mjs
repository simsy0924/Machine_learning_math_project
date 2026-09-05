#!/usr/bin/env node
// Reproducible training-speed benchmark for the math block workspace.
//
// It serves the repository over localhost, drives the real page in headless
// Chromium, builds a fixed training graph out of ordinary blocks plus
// user-authored manual backward definitions, and runs it. Two numbers come
// back:
//
//   - step/s   : how fast the training repeat runs
//   - checksum : a bit-exact hash of the trained weights
//
// The checksum is the safety net for optimization work. Any change that only
// removes overhead must leave it byte-for-byte identical.
//
// There is no automatic differentiation here: every gradient below is a
// backward formula written out of the same math blocks the palette offers, in
// an execution order this file states explicitly. That is exactly what the
// workspace asks a user to do, so the benchmark measures the real hot path.

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
    // Plain per-sample SGD over a strict cat/fish/house cycle. At lr 0.05 the
    // weights oscillate enough that the final model can score worse on held-out
    // samples than the untrained one, which makes "did it learn?" unanswerable.
    // 0.01 keeps the endpoint stable without changing what is measured.
    lr: 0.01,
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
  '.wasm': 'application/wasm',
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

  const SOFTMAX_CE_ID = 'bench-softmax-cross-entropy';
  const inputSource = manualBackpropInputSourceId;

  // ---------- backward formulas, written by hand ----------

  // Each definition below is the same JSON the 역전파 직접 정의 editor saves. It
  // is ordinary math built from palette blocks; nothing derives it.
  function installManualBackwardDefinitions() {
    // y = a + b  ->  da = g·1, db = g·1
    MANUAL_BACKPROP_BUILTINS.set('add', {
      nodes: [
        { id: 1, type: 'number', params: { value: 1 } },
        { id: 2, type: 'multiply', params: {} }
      ],
      connections: [
        { from: MANUAL_BACKPROP_UPSTREAM_ID, to: 2, inputIndex: 0 },
        { from: 1, to: 2, inputIndex: 1 }
      ],
      gradientOutputNodeIds: [2, 2]
    });

    // y = Ax  ->  dA = g ⊗ x, dx = Aᵀg
    MANUAL_BACKPROP_BUILTINS.set('matvec', {
      nodes: [
        { id: 1, type: 'outerProduct', params: {} },
        { id: 2, type: 'transposeMatvec', params: {} }
      ],
      connections: [
        { from: MANUAL_BACKPROP_UPSTREAM_ID, to: 1, inputIndex: 0 },
        { from: inputSource(1), to: 1, inputIndex: 1 },
        { from: inputSource(0), to: 2, inputIndex: 0 },
        { from: MANUAL_BACKPROP_UPSTREAM_ID, to: 2, inputIndex: 1 }
      ],
      gradientOutputNodeIds: [1, 2]
    });

    // y = max(a, b) with a = 0  ->  db = g·[y = b], the hand-written ReLU rule.
    // It reads the forward output y, so this definition also exercises the
    // "backward needs the owner's forward value" path.
    MANUAL_BACKPROP_BUILTINS.set('maximum', {
      nodes: [
        { id: 1, type: 'equal', params: {} },
        { id: 2, type: 'multiply', params: {} }
      ],
      connections: [
        { from: MANUAL_BACKPROP_OUTPUT_ID, to: 1, inputIndex: 0 },
        { from: inputSource(1), to: 1, inputIndex: 1 },
        { from: MANUAL_BACKPROP_UPSTREAM_ID, to: 2, inputIndex: 0 },
        { from: 1, to: 2, inputIndex: 1 }
      ],
      gradientOutputNodeIds: [2, 2]
    });
  }

  // Softmax + cross entropy as a 사용자 블록, with its own manual backward.
  //
  // Grouping in the UI keeps one external input port per internal input slot,
  // so z arrives twice: once for max(z) and once for z - max(z). The backward
  // formula recomputes p exactly as the forward graph does, which is what lets
  // the executor reuse the forward intermediates instead of redoing them.
  function installSoftmaxCrossEntropyBlock() {
    USER_BLOCKS.set(SOFTMAX_CE_ID, {
      id: SOFTMAX_CE_ID,
      name: 'Softmax CE',
      nodes: [
        { id: 1, type: 'arrayMax', params: {} },
        { id: 2, type: 'subtract', params: {} },
        { id: 3, type: 'exp', params: {} },
        { id: 4, type: 'sum', params: {} },
        { id: 5, type: 'divide', params: {} },
        { id: 6, type: 'log', params: {} },
        { id: 7, type: 'multiply', params: {} },
        { id: 8, type: 'sum', params: {} },
        { id: 9, type: 'number', params: { value: -1 } },
        { id: 10, type: 'multiply', params: {} }
      ],
      connections: [
        { from: 1, to: 2, inputIndex: 1 },
        { from: 2, to: 3, inputIndex: 0 },
        { from: 3, to: 4, inputIndex: 0 },
        { from: 3, to: 5, inputIndex: 0 },
        { from: 4, to: 5, inputIndex: 1 },
        { from: 5, to: 6, inputIndex: 0 },
        { from: 6, to: 7, inputIndex: 1 },
        { from: 7, to: 8, inputIndex: 0 },
        { from: 8, to: 10, inputIndex: 0 },
        { from: 9, to: 10, inputIndex: 1 }
      ],
      externalInputs: [
        { nodeId: 1, inputIndex: 0, label: 'z' },
        { nodeId: 2, inputIndex: 0, label: 'z' },
        { nodeId: 7, inputIndex: 0, label: 'y' }
      ],
      outputNodeId: 10,
      formula: 'L = -Σ y·ln(softmax(z))',
      // dL/dz = softmax(z) - y, derived by hand and written as blocks.
      manualBackprop: {
        nodes: [
          { id: 1, type: 'arrayMax', params: {} },
          { id: 2, type: 'subtract', params: {} },
          { id: 3, type: 'exp', params: {} },
          { id: 4, type: 'sum', params: {} },
          { id: 5, type: 'divide', params: {} },
          { id: 6, type: 'subtract', params: {} }
        ],
        connections: [
          { from: inputSource(0), to: 1, inputIndex: 0 },
          { from: inputSource(1), to: 2, inputIndex: 0 },
          { from: 1, to: 2, inputIndex: 1 },
          { from: 2, to: 3, inputIndex: 0 },
          { from: 3, to: 4, inputIndex: 0 },
          { from: 3, to: 5, inputIndex: 0 },
          { from: 4, to: 5, inputIndex: 1 },
          { from: 5, to: 6, inputIndex: 0 },
          { from: inputSource(2), to: 6, inputIndex: 1 }
        ],
        // Only the z gradient is ever stored; the other two destinations stay
        // blank in the graph below, so those branches never execute.
        gradientOutputNodeIds: [6, 6, 6]
      }
    });
  }

  function buildTrainingGraph() {
    resetWorkspace();
    installManualBackwardDefinitions();
    installSoftmaxCrossEntropyBlock();

    const add = (type, params) => {
      const node = addBlock(type);
      if (params) Object.assign(node.params, params);
      return node;
    };
    const link = (from, to, inputIndex = 0) => {
      graph.connections.push({ from: from.id, to: to.id, inputIndex });
    };

    const classCount = classes.length;

    // sample -> x, y
    const dataset = add('dataset');
    const index = add('variable', { name: 'i', mode: 'scalar', value: 0 });
    const sample = add('datasetSampleByIndex');
    link(dataset, sample, 0);
    link(index, sample, 1);
    const image = add('sampleImage');
    link(sample, image, 0);
    const x = add('flatten');
    link(image, x, 0);
    const target = add('sampleOneHot');
    link(sample, target, 0);

    // One affine layer: Wx + b. Both the forward nodes and the parameters are
    // returned so the backward chain can be wired to the very same values.
    const layers = [];
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

      const layer = { name, weights, bias, input, product, sum };
      layers.push(layer);
      return layer;
    };

    // With --hidden N the graph becomes a two-layer network with a ReLU built
    // from the ordinary 최댓값 block, which is what the project is actually for.
    let features = x;
    let inputSize = 784;
    let relu = null;
    let zero = null;
    if (hidden > 0) {
      const first = affine(x, '1', hidden, 784, seed);
      zero = add('number', { value: 0 });
      relu = add('maximum');
      link(zero, relu, 0);
      link(first.sum, relu, 1);
      features = relu;
      inputSize = hidden;
    }

    const last = affine(features, hidden > 0 ? '2' : '', classCount, inputSize, hidden > 0 ? seed + 1 : seed);
    const z = last.sum;

    const loss = add(`custom:${SOFTMAX_CE_ID}`);
    link(z, loss, 0);
    link(z, loss, 1);
    link(target, loss, 2);

    // Every gradient the backward chain writes needs a 변수 block to name it.
    const gradientNames = ['gz', 'gh', ...layers.flatMap(layer => [
      `gz${layer.name}`, `gp${layer.name}`, `gW${layer.name}`, `gb${layer.name}`
    ])];
    for (const name of gradientNames) add('variable', { name, mode: 'scalar', value: 0 });
    add('variable', { name: 'gL', mode: 'scalar', value: 1 });

    // ---------- backward pass, in the order this file chooses ----------
    const backward = [];
    const backwardBlock = (type, params, inputs) => {
      const node = add(type, { manualBackpropMode: 'backward', ...params });
      inputs.forEach((from, inputIndex) => link(from, node, inputIndex));
      backward.push(node);
      return node;
    };

    // dL/dz = p - y. Only the second z port stores a gradient.
    backwardBlock(`custom:${SOFTMAX_CE_ID}`, {
      manualBackpropUpstream: 'gL',
      manualBackpropGradient0: '',
      manualBackpropGradient1: 'gz',
      manualBackpropGradient2: ''
    }, [z, z, target]);

    let upstream = 'gz';
    for (let i = layers.length - 1; i >= 0; i--) {
      const layer = layers[i];
      const isFirst = i === 0;

      // z = Wx + b  ->  dW-side gradient and db
      backwardBlock('add', {
        manualBackpropUpstream: upstream,
        manualBackpropGradient0: `gp${layer.name}`,
        manualBackpropGradient1: `gb${layer.name}`
      }, [layer.product, layer.bias]);

      // p = Wx  ->  dW = g ⊗ x, dx = Wᵀg. The first layer's dx would be the
      // image gradient: its destination stays blank, so that branch is skipped.
      backwardBlock('matvec', {
        manualBackpropUpstream: `gp${layer.name}`,
        manualBackpropGradient0: `gW${layer.name}`,
        manualBackpropGradient1: isFirst ? '' : 'gh'
      }, [layer.weights, layer.input]);

      if (!isFirst) {
        // h = max(0, z1)  ->  dz1 = g·[h = z1]
        backwardBlock('maximum', {
          manualBackpropUpstream: 'gh',
          manualBackpropGradient0: '',
          manualBackpropGradient1: `gz${layers[i - 1].name}`
        }, [zero, layers[i - 1].sum]);
        upstream = `gz${layers[i - 1].name}`;
      }
    }

    // ---------- SGD update ----------
    const learningRate = add('number', { value: lr });
    const updates = [];
    for (const layer of layers) {
      for (const [parameter, gradientName] of [[layer.weights, `gW${layer.name}`], [layer.bias, `gb${layer.name}`]]) {
        const gradient = add('variable', { name: gradientName, mode: 'scalar', value: 0 });
        const scaled = add('multiply');
        link(learningRate, scaled, 0);
        link(gradient, scaled, 1);
        const updated = add('subtract');
        link(parameter, updated, 0);
        link(scaled, updated, 1);
        const assign = add('setVariable', { variable: parameter.params.name });
        link(updated, assign, 0);
        updates.push(assign);
      }
    }

    // 둘 다 계산 takes two inputs, so chain it to pin the forward pass, then
    // every backward block in order, then the parameter updates.
    let body = loss;
    for (const node of [...backward, ...updates]) {
      const sequence = add('sequence');
      link(body, sequence, 0);
      link(node, sequence, 1);
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
  // The window is fixed rather than derived from --steps, so the same samples
  // are scored no matter how long the run is and the before/after numbers stay
  // comparable between invocations.
  const HELD_OUT_FIRST_SAMPLE = 9500;

  function heldOutLoss(lossId, indexNode, sampleCount = 300) {
    const perClass = Math.floor(steps / classes.length);
    if (perClass >= HELD_OUT_FIRST_SAMPLE) return null;

    const signature = variableSignature(indexNode);
    const first = HELD_OUT_FIRST_SAMPLE * classes.length;
    let total = 0;

    for (let i = 0; i < sampleCount; i++) {
      RUNTIME_VARIABLES.set('i', { value: first + i, signature });
      total += evaluateNode(lossId, new Map(), new Set());
    }
    return total / sampleCount;
  }

  // A block that threw leaves its message on the node. Without this check a
  // broken graph still reports a step rate, because the repeat aborts on its
  // first iteration and the wall clock simply looks fast.
  function collectGraphErrors() {
    const errors = [];
    for (const node of graph.nodes.values()) {
      if (node.lastError) errors.push(`${node.type}: ${node.lastError}`);
    }
    return errors;
  }

  // Every run must start from the same state or the checksums diverge.
  function resetRuntimeState() {
    RUNTIME_VARIABLES.clear();
    pendingVariableUpdates = null;
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

    // Loss before any training, so the report can prove the graph learned.
    resetRuntimeState();
    const untrainedChecksum = weightChecksum();
    const untrainedLoss = heldOutLoss(loss.id, index);

    const results = [];

    // The first runs are discarded: the JIT has not warmed up on the compiled
    // execution plans yet and they read 30-40% slow.
    for (let run = 0; run < warmup + runs; run++) {
      resetRuntimeState();
      const started = performance.now();
      await evaluateGraph();
      const elapsedMs = performance.now() - started;

      const errors = collectGraphErrors();
      if (errors.length) throw new Error(`그래프 오류:\n  ${errors.join('\n  ')}`);
      if (run < warmup) continue;

      results.push({
        elapsedMs,
        stepsPerSecond: steps / (elapsedMs / 1000),
        msPerStep: elapsedMs / steps,
        checksum: weightChecksum(),
        heldOutLoss: heldOutLoss(loss.id, index)
      });
    }

    if (results[0].checksum === untrainedChecksum) {
      throw new Error('학습 후 가중치가 초기값과 같습니다. 학습이 실행되지 않았습니다.');
    }

    return { classes: loaded, steps, warmup, hidden, untrainedLoss, results };
  })();
}

// ---------- driver ----------

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const chromium = loadChromium();
  const { server, port } = await startServer();
  const browser = await chromium.launch({ headless: !options.headed, executablePath: process.env.CHROMIUM_EXECUTABLE_PATH || undefined });

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

    const trainedLoss = report.results[0].heldOutLoss;
    if (report.untrainedLoss != null && trainedLoss != null && trainedLoss >= report.untrainedLoss) {
      console.error('\n경고: held-out 손실이 줄지 않았습니다. 역전파가 실제로 학습하고 있지 않습니다.');
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
  if (loss != null) {
    const before = report.untrainedLoss != null ? ` (학습 전 ${report.untrainedLoss.toFixed(6)})` : '';
    console.log(`  학습 후 held-out 손실 ${loss.toFixed(6)}${before}`);
  }

  if (failures.length) {
    console.log('\n페이지 오류:');
    for (const failure of failures) console.log(`  ${failure}`);
  }
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
