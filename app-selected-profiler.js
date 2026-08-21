// Low-overhead sampled profiler for selected bulk evaluation.
// Only every 64th compiled leaf step is timed in detail so the profiler itself
// does not materially distort a long 5,000-step test.

(function installSelectedEvaluationProfiler() {
  const SAMPLE_EVERY = 64;
  let active = false;
  let sampleActive = false;
  let leafCallCount = 0;
  let profile = null;
  let lastProfile = null;

  function blockLabel(type) {
    try {
      return getBlockDef(type)?.title || type;
    } catch {
      return type;
    }
  }

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

  function recordNode(node, elapsedMs) {
    if (!profile || !sampleActive) return;
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

  // Wrap ordinary block compute functions. The wrappers are nearly free when a
  // sample is not active; performance.now() is called only on sampled steps.
  for (const [type, def] of Object.entries(BLOCKS)) {
    if (typeof def?.compute !== 'function') continue;
    const originalCompute = def.compute;
    def.compute = function profiledBlockCompute(node, inputs) {
      if (!active || !sampleActive) return originalCompute(node, inputs);
      const started = performance.now();
      try {
        return originalCompute(node, inputs);
      } finally {
        recordNode(node || { type }, performance.now() - started);
      }
    };
  }

  // Count compiled leaf executions and sample every Nth one. This catches the
  // actual selected-test hot path without changing the graph or its math.
  const executeSelectedLeafPlanBeforeProfile = executeSelectedLeafPlan;
  executeSelectedLeafPlan = function profiledSelectedLeafPlan(plan) {
    if (!active || !profile) return executeSelectedLeafPlanBeforeProfile(plan);

    const shouldSample = (leafCallCount++ % SAMPLE_EVERY) === 0;
    if (!shouldSample) return executeSelectedLeafPlanBeforeProfile(plan);

    sampleActive = true;
    const started = performance.now();
    try {
      return executeSelectedLeafPlanBeforeProfile(plan);
    } finally {
      profile.sampledLeafMs += performance.now() - started;
      profile.sampledSteps++;
      sampleActive = false;
    }
  };

  // Reset/finalize around one selected stateful calculation.
  const evaluateSelectedStatefulBranchBeforeProfile = evaluateSelectedStatefulBranch;
  evaluateSelectedStatefulBranch = async function profiledSelectedStatefulBranch(nodeId) {
    const outermost = !active;
    if (outermost) {
      active = true;
      sampleActive = false;
      leafCallCount = 0;
      profile = newProfile();
    }

    const started = outermost ? performance.now() : 0;
    try {
      const result = await evaluateSelectedStatefulBranchBeforeProfile(nodeId);
      if (outermost && profile) {
        profile.wallMs = performance.now() - started;
        profile.completedSteps = Number(result?.progress?.completed) || leafCallCount;
      }
      return result;
    } finally {
      if (outermost) {
        lastProfile = profile;
        profile = null;
        active = false;
        sampleActive = false;
      }
    }
  };

  function appendProfileResult() {
    const target = selectedResultElement?.();
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

    const entries = Array.from(p.byType.values())
      .sort((a, b) => b.ms - a.ms)
      .slice(0, 8);

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

  const renderSelectedEvaluationResultBeforeProfile = renderSelectedEvaluationResult;
  renderSelectedEvaluationResult = function profiledSelectedResult(value) {
    renderSelectedEvaluationResultBeforeProfile(value);
    appendProfileResult();
  };
})();
