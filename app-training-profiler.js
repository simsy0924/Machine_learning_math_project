// Sampled profiler for the main training path.
// Diagnostic only: it does not change math, gradients, or update order.

(function installTrainingProfiler() {
  const SAMPLE_EVERY = 64;
  let run = null;
  let gradientCallCount = 0;
  let sampleStepActive = false;
  let sampleStepStart = 0;
  let insideGradient = false;

  function makeBucketMap() {
    return new Map();
  }

  function record(map, key, ms) {
    let bucket = map.get(key);
    if (!bucket) {
      bucket = { ms: 0, calls: 0 };
      map.set(key, bucket);
    }
    bucket.ms += ms;
    bucket.calls++;
  }

  function ensurePanel() {
    let panel = document.getElementById('trainingProfilerPanel');
    if (panel) return panel;

    const inspector = document.querySelector('.inspector-panel');
    if (!inspector) return null;

    panel = document.createElement('section');
    panel.className = 'inspector-card';
    panel.id = 'trainingProfilerPanel';
    panel.innerHTML = `
      <div class="section-head"><h2>학습 계산 프로파일</h2></div>
      <div id="trainingProfilerResult" class="dataset-status">계산 버튼으로 학습을 실행하면 병목을 샘플링해서 보여줍니다.</div>`;
    inspector.insertBefore(panel, inspector.children[1] || null);
    return panel;
  }

  function resultElement() {
    ensurePanel();
    return document.getElementById('trainingProfilerResult');
  }

  function resetRun() {
    gradientCallCount = 0;
    sampleStepActive = false;
    insideGradient = false;
    run = {
      active: true,
      sawDisabled: false,
      startedAt: performance.now(),
      completed: 0,
      total: 0,
      sampledSteps: 0,
      sampledStepMs: 0,
      sampledGradientPasses: 0,
      sampledGradientMs: 0,
      forwardByType: makeBucketMap(),
      outsideByType: makeBucketMap(),
      backwardByType: makeBucketMap(),
      kernels: makeBucketMap()
    };

    const target = resultElement();
    if (target) target.textContent = `학습 실행 중 · ${SAMPLE_EVERY} step마다 1회 샘플링`;
  }

  function closeSampleStep(now = performance.now()) {
    if (!run?.active || !sampleStepActive) return;
    run.sampledStepMs += Math.max(0, now - sampleStepStart);
    run.sampledSteps++;
    sampleStepActive = false;
    insideGradient = false;
  }

  function sortedBuckets(map) {
    return Array.from(map.entries()).sort((a, b) => b[1].ms - a[1].ms);
  }

  function blockLabel(type) {
    return BLOCKS[type]?.title || type;
  }

  function formatBucketLines(title, map, divisor, limit = 8) {
    const rows = sortedBuckets(map).slice(0, limit);
    if (!rows.length || divisor <= 0) return [];
    const total = rows.reduce((sum, [, bucket]) => sum + bucket.ms, 0);
    const lines = [title];
    rows.forEach(([key, bucket], index) => {
      const per = bucket.ms / divisor;
      const share = total > 0 ? bucket.ms / total * 100 : 0;
      lines.push(`${index + 1}. ${blockLabel(key)} · ${per.toFixed(3)} ms/sample · ${share.toFixed(1)}% · ${bucket.calls}회`);
    });
    return lines;
  }

  function renderRun() {
    const target = resultElement();
    if (!target || !run) return;

    const completed = run.completed || gradientCallCount;
    const wallMs = Math.max(0, performance.now() - run.startedAt);
    const wallPerStep = completed > 0 ? wallMs / completed : 0;
    const sampledStepPer = run.sampledSteps > 0 ? run.sampledStepMs / run.sampledSteps : 0;
    const gradientPer = run.sampledGradientPasses > 0 ? run.sampledGradientMs / run.sampledGradientPasses : 0;
    const outsideGradient = Math.max(0, sampledStepPer - gradientPer);
    const measuredRate = wallPerStep > 0 ? 1000 / wallPerStep : 0;

    const lines = [
      `샘플 간격 ${SAMPLE_EVERY} step · 샘플 ${run.sampledSteps.toLocaleString('ko-KR')}회`,
      `완료 step ${completed.toLocaleString('ko-KR')}${run.total ? ` / ${run.total.toLocaleString('ko-KR')}` : ''}`,
      `전체 벽시계 ${wallPerStep.toFixed(3)} ms/step · 약 ${Math.round(measuredRate).toLocaleString('ko-KR')} step/s`,
      `샘플 step ${sampledStepPer.toFixed(3)} ms/step`,
      `gradient 전체 ${gradientPer.toFixed(3)} ms/step`,
      `gradient 밖 비용(대략) ${outsideGradient.toFixed(3)} ms/step`
    ];

    const kernelRows = sortedBuckets(run.kernels);
    if (kernelRows.length && run.sampledSteps > 0) {
      lines.push('', '핵심 수치 커널');
      kernelRows.forEach(([name, bucket], index) => {
        lines.push(`${index + 1}. ${name} · ${(bucket.ms / run.sampledSteps).toFixed(3)} ms/sample · ${bucket.calls}회`);
      });
    }

    lines.push('', ...formatBucketLines('gradient forward 블록', run.forwardByType, run.sampledGradientPasses));
    lines.push('', ...formatBucketLines('gradient backward VJP', run.backwardByType, run.sampledGradientPasses));
    lines.push('', ...formatBucketLines('gradient 밖 블록', run.outsideByType, run.sampledSteps));

    const pre = document.createElement('pre');
    pre.style.whiteSpace = 'pre-wrap';
    pre.style.margin = '0';
    pre.style.fontFamily = 'inherit';
    pre.textContent = lines.filter((line, i, arr) => !(line === '' && arr[i - 1] === '')).join('\n');
    target.replaceChildren(pre);
  }

  function finishRun() {
    if (!run?.active) return;
    closeSampleStep();
    run.active = false;
    renderRun();
  }

  ensurePanel();

  // Start before the existing Calculate listener runs.
  evaluateBtn.addEventListener('click', () => {
    if (!evaluateBtn.disabled) resetRun();
  }, true);

  // The existing async Calculate handler toggles the disabled attribute for the
  // entire run. Observe it instead of replacing the handler reference that was
  // already registered by app-boot.js.
  const buttonObserver = new MutationObserver(mutations => {
    if (!run?.active) return;
    for (const mutation of mutations) {
      if (mutation.attributeName !== 'disabled') continue;
      if (mutation.oldValue == null) run.sawDisabled = true;
    }
    if (run.sawDisabled && !evaluateBtn.hasAttribute('disabled')) finishRun();
  });
  buttonObserver.observe(evaluateBtn, {
    attributes: true,
    attributeFilter: ['disabled'],
    attributeOldValue: true
  });

  // Keep the progress counters so the final wall-time rate can be compared with
  // the live step/s shown in the workspace.
  const maybeYieldProgressBeforeTrainingProfiler = maybeYieldProgress;
  maybeYieldProgress = async function(progress, force = false) {
    if (run?.active && progress) {
      run.completed = Number(progress.completed) || 0;
      run.total = Number(progress.total) || 0;
    }
    return await maybeYieldProgressBeforeTrainingProfiler(progress, force);
  };

  // One computeAllGraphGradients call corresponds to one fresh SGD state in the
  // current shared-gradient design. Use its call boundary as the sampled step
  // boundary. The interval from one gradient start to the next includes the rest
  // of the update plus the next step's pre-gradient work.
  const computeAllGraphGradientsBeforeTrainingProfiler = computeAllGraphGradients;
  computeAllGraphGradients = function(outputId) {
    if (!run?.active) return computeAllGraphGradientsBeforeTrainingProfiler(outputId);

    const now = performance.now();
    closeSampleStep(now);
    gradientCallCount++;

    const shouldSample = ((gradientCallCount - 1) % SAMPLE_EVERY) === 0;
    if (shouldSample) {
      sampleStepActive = true;
      sampleStepStart = now;
      insideGradient = true;
    }

    const started = shouldSample ? performance.now() : 0;
    try {
      return computeAllGraphGradientsBeforeTrainingProfiler(outputId);
    } finally {
      if (shouldSample && run?.active) {
        run.sampledGradientMs += performance.now() - started;
        run.sampledGradientPasses++;
      }
      insideGradient = false;
    }
  };

  // Forward block time inside the sampled gradient pass, and ordinary block work
  // outside the gradient pass but inside the sampled SGD-step interval.
  for (const [type, def] of Object.entries(BLOCKS)) {
    if (typeof def.compute !== 'function') continue;
    const original = def.compute;
    def.compute = function(node, inputs) {
      if (!run?.active || !sampleStepActive) return original(node, inputs);
      const started = performance.now();
      try {
        return original(node, inputs);
      } finally {
        record(insideGradient ? run.forwardByType : run.outsideByType, type, performance.now() - started);
      }
    };
  }

  // Reverse-mode cost grouped by mathematical primitive type.
  const primitiveVJPBeforeTrainingProfiler = primitiveVJP;
  primitiveVJP = function(type, inputs, output, upstream) {
    if (!run?.active || !sampleStepActive || !insideGradient) {
      return primitiveVJPBeforeTrainingProfiler(type, inputs, output, upstream);
    }
    const started = performance.now();
    try {
      return primitiveVJPBeforeTrainingProfiler(type, inputs, output, upstream);
    } finally {
      record(run.backwardByType, type, performance.now() - started);
    }
  };

  function wrapKernel(name, getter, setter) {
    const original = getter();
    setter(function(...args) {
      if (!run?.active || !sampleStepActive) return original(...args);
      const started = performance.now();
      try {
        return original(...args);
      } finally {
        record(run.kernels, name, performance.now() - started);
      }
    });
  }

  wrapKernel('행렬 × 벡터', () => matrixVectorValues, fn => { matrixVectorValues = fn; });
  wrapKernel('전치 행렬 × 벡터', () => transposeMatrixVectorValues, fn => { transposeMatrixVectorValues = fn; });
  wrapKernel('외적', () => outerProductValues, fn => { outerProductValues = fn; });
})();
