// Sampled profilers for the two long-running paths: 계산 (training) and
// 선택 계산 (bulk evaluation).
//
// Automatic differentiation no longer exists, so the training profiler measures
// the thing that actually represents one learning step now: one compiled leaf
// 반복 iteration. Every 64th iteration is timed in detail, including each plan
// node, regardless of whether that node is forward math, hand-written backward
// math, a newly added transform, a custom block, a variable read or an update.

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

  function record(map, key, ms, extra = null) {
    let bucket = map.get(key);
    if (!bucket) {
      bucket = { ms: 0, calls: 0, ...(extra || {}) };
      map.set(key, bucket);
    }
    bucket.ms += ms;
    bucket.calls++;
    return bucket;
  }

  function sortedBuckets(map) {
    return Array.from(map.entries()).sort((a, b) => b[1].ms - a[1].ms);
  }

  // ================= 학습 계산 프로파일 =================

  let run = null;
  let trainingLeafCallCount = 0;
  let trainingSampleActive = false;

  const trainingSampling = () => Boolean(run?.active && trainingSampleActive);

  function trainingResultElement() {
    inspectorPanel(
      'trainingProfilerPanel',
      '학습 계산 프로파일',
      '계산 버튼으로 반복 학습을 실행하면 실제 수학 블록 병목을 샘플링해서 보여줍니다.'
    );
    return document.getElementById('trainingProfilerPanelResult');
  }

  function resetRun() {
    trainingLeafCallCount = 0;
    trainingSampleActive = false;
    run = {
      active: true,
      sawDisabled: false,
      startedAt: performance.now(),
      completed: 0,
      total: 0,
      sampledSteps: 0,
      sampledStepMs: 0,
      sampledNodeMs: 0,
      byType: new Map(),
      byNode: new Map(),
      kernels: new Map()
    };

    const target = trainingResultElement();
    if (target) target.textContent = `학습 실행 중 · ${SAMPLE_EVERY} step마다 1회 샘플링`;
  }

  function recordTrainingNode(type, node, ms) {
    if (!run) return;
    run.sampledNodeMs += ms;

    const key = String(type || node?.type || 'unknown');
    const typeEntry = record(run.byType, key, ms, { type: key, nodeIds: new Set() });
    if (node?.id != null) typeEntry.nodeIds.add(node.id);

    const nodeKey = `${node?.id ?? '?'}:${key}`;
    record(run.byNode, nodeKey, ms, {
      id: node?.id,
      type: key,
      backward: node?.params?.manualBackpropMode === 'backward'
    });
  }

  function formatTrainingTypeLines(divisor, limit = 10) {
    if (divisor <= 0) return [];
    const rows = sortedBuckets(run.byType).slice(0, limit);
    if (!rows.length) return [];
    const measured = Array.from(run.byType.values()).reduce((sum, bucket) => sum + bucket.ms, 0);
    const lines = ['느린 블록 종류'];
    rows.forEach(([, bucket], index) => {
      const per = bucket.ms / divisor;
      const share = measured > 0 ? bucket.ms / measured * 100 : 0;
      const count = bucket.nodeIds?.size || 0;
      lines.push(`${index + 1}. ${blockLabel(bucket.type)}${count > 1 ? ` (${count}개)` : ''} · ${per.toFixed(3)} ms/step · 노드 시간 ${share.toFixed(1)}%`);
    });
    return lines;
  }

  function formatTrainingNodeLines(divisor, limit = 10) {
    if (divisor <= 0) return [];
    const rows = sortedBuckets(run.byNode).slice(0, limit);
    if (!rows.length) return [];
    const lines = ['느린 개별 블록'];
    rows.forEach(([, bucket], index) => {
      const suffix = bucket.backward ? ' · 역전파 실행' : '';
      lines.push(`${index + 1}. #${bucket.id ?? '?'} ${blockLabel(bucket.type)}${suffix} · ${(bucket.ms / divisor).toFixed(3)} ms/step`);
    });
    return lines;
  }

  function renderRun() {
    const target = trainingResultElement();
    if (!target || !run) return;

    const completed = run.completed || trainingLeafCallCount;
    const wallMs = Math.max(0, performance.now() - run.startedAt);
    const wallPerStep = completed > 0 ? wallMs / completed : 0;
    const sampledStepPer = run.sampledSteps > 0 ? run.sampledStepMs / run.sampledSteps : 0;
    const sampledNodePer = run.sampledSteps > 0 ? run.sampledNodeMs / run.sampledSteps : 0;
    const insideOther = Math.max(0, sampledStepPer - sampledNodePer);
    const outsideLeaf = Math.max(0, wallPerStep - sampledStepPer);
    const measuredRate = wallPerStep > 0 ? 1000 / wallPerStep : 0;

    const lines = [
      `샘플 간격 ${SAMPLE_EVERY} step · 샘플 ${run.sampledSteps.toLocaleString('ko-KR')}회`,
      `완료 step ${completed.toLocaleString('ko-KR')}${run.total ? ` / ${run.total.toLocaleString('ko-KR')}` : ''}`,
      `전체 벽시계 ${wallPerStep.toFixed(3)} ms/step · 약 ${Math.round(measuredRate).toLocaleString('ko-KR')} step/s`,
      `샘플 반복 실행 ${sampledStepPer.toFixed(3)} ms/step`,
      `노드 실행 합 ${sampledNodePer.toFixed(3)} ms/step`,
      `step 내부 기타 비용 ${insideOther.toFixed(3)} ms/step`,
      `반복 제어·상태 반영·UI 등 ${outsideLeaf.toFixed(3)} ms/step`
    ];

    const kernelRows = sortedBuckets(run.kernels);
    if (kernelRows.length && run.sampledSteps > 0) {
      lines.push('', '핵심 수치 커널');
      kernelRows.slice(0, 10).forEach(([name, bucket], index) => {
        lines.push(`${index + 1}. ${name} · ${(bucket.ms / run.sampledSteps).toFixed(3)} ms/step · ${bucket.calls}회`);
      });
    }

    lines.push('', ...formatTrainingTypeLines(run.sampledSteps));
    lines.push('', ...formatTrainingNodeLines(run.sampledSteps));

    if (!run.sampledSteps && completed > 0) {
      lines.push('', `상세 샘플이 없습니다. ${SAMPLE_EVERY} step보다 짧은 반복이거나 컴파일되지 않은 경로일 수 있습니다.`);
    }

    const pre = document.createElement('pre');
    pre.style.whiteSpace = 'pre-wrap';
    pre.style.margin = '0';
    pre.style.fontFamily = 'inherit';
    pre.textContent = lines.filter((line, i, arr) => !(line === '' && arr[i - 1] === '')).join('\n');
    target.replaceChildren(pre);
  }

  function finishRun() {
    if (!run?.active) return;
    trainingSampleActive = false;
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
  let selectedLeafCallCount = 0;
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
    if (!profile) return;
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
    summary.textContent = `${SAMPLE_EVERY} step 간격 ${p.sampledSteps}회 샘플 · 전체 ${wallPerStep.toFixed(3)} ms/step · 샘플 순수 실행 ${leafPerStep.toFixed(3)} ms/step · 노드 compute 합 ${nodePerStep.toFixed(3)} ms/step`;
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

  // The old per-block wrapper remains useful for 선택 계산's interpreted path.
  // Training is timed at the compiled-plan step level instead, so newly added
  // blocks and variable/update steps cannot disappear from the profile.
  PROFILER_HOOKS.blockCompute = {
    active: selectedSampling,
    record(type, node, ms) {
      if (selectedSampling()) recordSelectedNode(node || { type }, ms);
    }
  };

  PROFILER_HOOKS.kernel = {
    active: trainingSampling,
    record: (name, ms) => {
      if (run) record(run.kernels, name, ms);
    }
  };

  PROFILER_HOOKS.leafNode = {
    active: mode => mode === 'training' && trainingSampling(),
    record: recordTrainingNode
  };

  PROFILER_HOOKS.leafStep = {
    begin(mode) {
      if (mode === 'training' && run?.active) {
        const index = trainingLeafCallCount++;
        if ((index % SAMPLE_EVERY) !== 0) return null;
        trainingSampleActive = true;
        return { kind: 'training', startedAt: performance.now() };
      }

      if (mode === 'selected' && selectedActive && profile) {
        const index = selectedLeafCallCount++;
        if ((index % SAMPLE_EVERY) !== 0) return null;
        selectedSampleActive = true;
        return { kind: 'selected', startedAt: performance.now() };
      }

      return null;
    },
    end(token) {
      if (!token) return;
      const elapsed = performance.now() - token.startedAt;
      if (token.kind === 'training') {
        if (run?.active) {
          run.sampledStepMs += elapsed;
          run.sampledSteps++;
        }
        trainingSampleActive = false;
      } else if (token.kind === 'selected') {
        if (profile) {
          profile.sampledLeafMs += elapsed;
          profile.sampledSteps++;
        }
        selectedSampleActive = false;
      }
    }
  };

  // Keep the progress counters so the final wall-time rate can be compared with
  // the live step/s shown in the workspace.
  PROFILER_HOOKS.progress = progressState => {
    if (!run?.active || !progressState) return;
    run.completed = Number(progressState.completed) || 0;
    run.total = Number(progressState.total) || 0;
  };

  PROFILER_HOOKS.selectedRun = {
    begin() {
      if (selectedActive) return null;
      selectedActive = true;
      selectedSampleActive = false;
      selectedLeafCallCount = 0;
      profile = newProfile();
      return { outermost: true, startedAt: performance.now() };
    },
    end(token, result) {
      if (!token?.outermost || !profile) return;
      profile.wallMs = performance.now() - token.startedAt;
      profile.completedSteps = Number(result?.progress?.completed) || selectedLeafCallCount;
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