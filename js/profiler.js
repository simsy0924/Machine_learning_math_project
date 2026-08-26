// Sampled profilers for the two long-running paths: 계산 (training) and
// 선택 계산 (bulk evaluation).
//
// Diagnostic only: nothing here changes math, gradients or update order. Both
// profilers work by filling in PROFILER_HOOKS, which the rest of the app checks
// for null on entry — so with this file removed the app behaves identically.
// Only every 64th step is timed in detail, so a long run is not distorted.

(function installProfilers() {
  const SAMPLE_EVERY = 64;

  function inspectorPanel(id, title, initialText) {
    let panel = document.getElementById(id);
    if (panel) return panel;

    const inspector = document.querySelector('.inspector-panel');
    if (!inspector) return null;

    panel = document.createElement('section');
    panel.className = 'inspector-card';
    panel.id = id;
    panel.innerHTML = `
      <div class="section-head"><h2>${title}</h2></div>
      <div id="${id}Result" class="dataset-status">${initialText}</div>`;
    inspector.insertBefore(panel, inspector.children[1] || null);
    return panel;
  }

  function blockLabel(type) {
    try {
      return getBlockDef(type)?.title || type;
    } catch {
      return type;
    }
  }

  // ================= 학습 계산 프로파일 =================

  let run = null;
  let gradientCallCount = 0;
  let sampleStepActive = false;
  let sampleStepStart = 0;
  let insideGradient = false;

  const trainingSampling = () => Boolean(run?.active && sampleStepActive);

  function record(map, key, ms) {
    let bucket = map.get(key);
    if (!bucket) {
      bucket = { ms: 0, calls: 0 };
      map.set(key, bucket);
    }
    bucket.ms += ms;
    bucket.calls++;
  }

  function trainingResultElement() {
    inspectorPanel('trainingProfilerPanel', '학습 계산 프로파일', '계산 버튼으로 학습을 실행하면 병목을 샘플링해서 보여줍니다.');
    return document.getElementById('trainingProfilerPanelResult');
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
      forwardByType: new Map(),
      outsideByType: new Map(),
      backwardByType: new Map(),
      kernels: new Map()
    };

    const target = trainingResultElement();
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
    const target = trainingResultElement();
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

  trainingResultElement();

  // Start before the existing Calculate listener runs.
  evaluateBtn.addEventListener('click', () => {
    if (!evaluateBtn.disabled) resetRun();
  }, true);

  // The async Calculate handler toggles the disabled attribute for the entire
  // run, so observing it needs no cooperation from the evaluator.
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

  // ================= 선택 계산 프로파일 =================

  let selectedActive = false;
  let selectedSampleActive = false;
  let leafCallCount = 0;
  let profile = null;
  let lastProfile = null;

  const selectedSampling = () => Boolean(selectedActive && selectedSampleActive && profile);

  function newProfile() {
    return {
      sampleEvery: SAMPLE_EVERY,
      sampledSteps: 0,
      sampledLeafMs: 0,
      sampledNodeMs: 0,
      wallMs: 0,
      completedSteps: 0,
      byType: new Map(),
      byNode: new Map()
    };
  }

  function recordSelectedNode(node, elapsedMs) {
    profile.sampledNodeMs += elapsedMs;

    const type = String(node?.type || 'unknown');
    const typeEntry = profile.byType.get(type) || { type, ms: 0, calls: 0, nodeIds: new Set() };
    typeEntry.ms += elapsedMs;
    typeEntry.calls++;
    if (node?.id != null) typeEntry.nodeIds.add(node.id);
    profile.byType.set(type, typeEntry);

    const nodeKey = `${node?.id ?? '?'}:${type}`;
    const nodeEntry = profile.byNode.get(nodeKey) || { id: node?.id, type, ms: 0, calls: 0 };
    nodeEntry.ms += elapsedMs;
    nodeEntry.calls++;
    profile.byNode.set(nodeKey, nodeEntry);
  }

  function appendSelectedProfileResult() {
    const target = selectedResultElement();
    const p = lastProfile;
    if (!target || !p || !p.sampledSteps || !p.completedSteps) return;

    const wallPerStep = p.wallMs / p.completedSteps;
    const leafPerStep = p.sampledLeafMs / p.sampledSteps;
    const nodePerStep = p.sampledNodeMs / p.sampledSteps;
    const outsidePerStep = Math.max(0, wallPerStep - leafPerStep);

    const wrapper = document.createElement('div');
    wrapper.style.marginTop = '14px';
    wrapper.style.paddingTop = '12px';
    wrapper.style.borderTop = '1px solid rgba(127,127,127,.35)';

    const title = document.createElement('strong');
    title.textContent = '선택 계산 프로파일';
    wrapper.appendChild(title);

    const summary = document.createElement('div');
    summary.style.marginTop = '6px';
    summary.style.fontSize = '12px';
    summary.style.lineHeight = '1.55';
    summary.textContent = `64 step 간격 ${p.sampledSteps}회 샘플 · 전체 ${wallPerStep.toFixed(3)} ms/step · 샘플 순수 실행 ${leafPerStep.toFixed(3)} ms/step · 노드 compute 합 ${nodePerStep.toFixed(3)} ms/step`;
    wrapper.appendChild(summary);

    const outside = document.createElement('div');
    outside.style.fontSize = '12px';
    outside.style.marginTop = '3px';
    outside.textContent = `반복/상태 반영/UI yield 등 노드 밖 비용(대략): ${outsidePerStep.toFixed(3)} ms/step`;
    wrapper.appendChild(outside);

    const entries = Array.from(p.byType.values()).sort((a, b) => b.ms - a.ms).slice(0, 8);
    if (entries.length) {
      const list = document.createElement('ol');
      list.style.margin = '8px 0 0';
      list.style.paddingLeft = '20px';
      list.style.fontSize = '12px';

      for (const entry of entries) {
        const item = document.createElement('li');
        const percent = p.sampledNodeMs > 0 ? (entry.ms / p.sampledNodeMs) * 100 : 0;
        const perStep = entry.ms / p.sampledSteps;
        const nodeCount = entry.nodeIds.size;
        item.textContent = `${blockLabel(entry.type)}${nodeCount > 1 ? ` (${nodeCount}개)` : ''} · 노드 시간 ${percent.toFixed(1)}% · ${perStep.toFixed(3)} ms/step`;
        list.appendChild(item);
      }
      wrapper.appendChild(list);
    }

    target.appendChild(wrapper);
  }

  SELECTED_RESULT_EXTENSIONS.push(appendSelectedProfileResult);

  // ================= hook installation =================

  PROFILER_HOOKS.blockCompute = {
    active: () => trainingSampling() || selectedSampling(),
    record(type, node, ms) {
      if (trainingSampling()) record(insideGradient ? run.forwardByType : run.outsideByType, type, ms);
      if (selectedSampling()) recordSelectedNode(node || { type }, ms);
    }
  };

  PROFILER_HOOKS.kernel = {
    active: trainingSampling,
    record: (name, ms) => record(run.kernels, name, ms)
  };

  // Reverse-mode cost grouped by mathematical primitive type.
  PROFILER_HOOKS.vjp = {
    active: () => trainingSampling() && insideGradient,
    record: (type, ms) => record(run.backwardByType, type, ms)
  };

  // One computeAllGraphGradients call corresponds to one fresh SGD state in the
  // current shared-gradient design, so its call boundary is the sampled step
  // boundary. The interval from one gradient start to the next includes the rest
  // of the update plus the next step's pre-gradient work.
  PROFILER_HOOKS.gradientPass = {
    begin() {
      if (!run?.active) return null;

      const now = performance.now();
      closeSampleStep(now);
      gradientCallCount++;

      const shouldSample = ((gradientCallCount - 1) % SAMPLE_EVERY) === 0;
      if (shouldSample) {
        sampleStepActive = true;
        sampleStepStart = now;
        insideGradient = true;
      }
      return shouldSample ? performance.now() : null;
    },
    end(started) {
      if (started != null && run?.active) {
        run.sampledGradientMs += performance.now() - started;
        run.sampledGradientPasses++;
      }
      if (run?.active) insideGradient = false;
    }
  };

  // Keep the progress counters so the final wall-time rate can be compared with
  // the live step/s shown in the workspace.
  PROFILER_HOOKS.progress = progress => {
    if (!run?.active || !progress) return;
    run.completed = Number(progress.completed) || 0;
    run.total = Number(progress.total) || 0;
  };

  PROFILER_HOOKS.selectedLeafStep = {
    begin() {
      if (!selectedActive || !profile) return null;
      if ((leafCallCount++ % SAMPLE_EVERY) !== 0) return null;
      selectedSampleActive = true;
      return performance.now();
    },
    end(started) {
      profile.sampledLeafMs += performance.now() - started;
      profile.sampledSteps++;
      selectedSampleActive = false;
    }
  };

  PROFILER_HOOKS.selectedRun = {
    begin() {
      if (selectedActive) return null;
      selectedActive = true;
      selectedSampleActive = false;
      leafCallCount = 0;
      profile = newProfile();
      return { outermost: true, startedAt: performance.now() };
    },
    end(token, result) {
      if (!token?.outermost || !profile) return;
      profile.wallMs = performance.now() - token.startedAt;
      profile.completedSteps = Number(result?.progress?.completed) || leafCallCount;
    },
    finish(token) {
      if (!token?.outermost) return;
      lastProfile = profile;
      profile = null;
      selectedActive = false;
      selectedSampleActive = false;
    }
  };
})();
