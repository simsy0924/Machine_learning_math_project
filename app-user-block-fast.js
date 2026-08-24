// Compiled execution for user-created blocks.
//
// Grouping blocks should be a visual abstraction, not a runtime penalty. The
// original user-block evaluator rebuilt Maps, searched connections, recursed
// through the internal graph, and then repeated the internal forward pass again
// during VJP. This file compiles each immutable user-block definition once.
//
// During a graph-gradient pass it also records the internal forward values in a
// small tape. userBlockVJP consumes those frames in reverse order, so the exact
// forward that produced a custom-block output is reused instead of recomputed.

(function installCompiledUserBlocks() {
  const PLAN_CACHE = new WeakMap();
  const MAX_FRAME_POOL = 32;

  let captureDepth = 0;
  let touchedPlans = new Set();

  function slotKey(nodeId, inputIndex) {
    return `${nodeId}:${inputIndex == null ? 'source' : inputIndex}`;
  }

  function compileUserBlockPlan(definition) {
    if (!definition || !Array.isArray(definition.nodes) || !Array.isArray(definition.connections)) {
      throw new Error('사용자 블록 정의가 올바르지 않습니다.');
    }

    const nodeById = new Map();
    for (const node of definition.nodes) nodeById.set(node.id, node);

    const externalIndexBySlot = new Map();
    const externalInputs = Array.isArray(definition.externalInputs) ? definition.externalInputs : [];
    for (let i = 0; i < externalInputs.length; i++) {
      const slot = externalInputs[i];
      const key = slotKey(slot.nodeId, slot.inputIndex);
      if (!externalIndexBySlot.has(key)) externalIndexBySlot.set(key, i);
    }

    // Preserve the previous Array.find semantics if a malformed imported block
    // contains duplicate connections: the first matching connection wins.
    const connectionByInput = new Map();
    for (const connection of definition.connections) {
      const key = slotKey(connection.to, connection.inputIndex);
      if (!connectionByInput.has(key)) connectionByInput.set(key, connection.from);
    }

    const order = [];
    const seen = new Set();
    const visiting = new Set();

    const visit = nodeId => {
      if (seen.has(nodeId)) return;
      if (visiting.has(nodeId)) throw new Error('사용자 블록 내부에 순환 연결이 있습니다.');

      const node = nodeById.get(nodeId);
      if (!node) throw new Error('사용자 블록 내부 노드를 찾을 수 없습니다.');

      visiting.add(nodeId);
      const sourceSlot = externalIndexBySlot.get(slotKey(nodeId, null));

      if (sourceSlot == null) {
        const blockDef = getBlockDef(node.type);
        if (blockDef.special === 'derivative') {
          throw new Error('현재 미분 블록 자체는 사용자 블록 내부에 넣을 수 없습니다.');
        }

        for (let inputIndex = 0; inputIndex < blockDef.inputs.length; inputIndex++) {
          const key = slotKey(nodeId, inputIndex);
          const fromNodeId = connectionByInput.get(key);
          if (fromNodeId != null) {
            visit(fromNodeId);
          } else if (!externalIndexBySlot.has(key)) {
            throw new Error(`사용자 블록 입력 '${blockDef.inputs[inputIndex]}'이 비어 있습니다.`);
          }
        }
      }

      visiting.delete(nodeId);
      seen.add(nodeId);
      order.push(nodeId);
    };

    visit(definition.outputNodeId);

    const valueIndexByNodeId = new Map();
    for (let i = 0; i < order.length; i++) valueIndexByNodeId.set(order[i], i);

    const steps = new Array(order.length);
    for (let valueIndex = 0; valueIndex < order.length; valueIndex++) {
      const nodeId = order[valueIndex];
      const node = nodeById.get(nodeId);
      const sourceSlot = externalIndexBySlot.get(slotKey(nodeId, null));

      if (sourceSlot != null) {
        steps[valueIndex] = {
          valueIndex,
          node,
          kind: 'externalSource',
          sourceSlot,
          inputRefs: [],
          inputs: []
        };
        continue;
      }

      const blockDef = getBlockDef(node.type);
      const inputRefs = new Array(blockDef.inputs.length);
      for (let inputIndex = 0; inputIndex < blockDef.inputs.length; inputIndex++) {
        const key = slotKey(nodeId, inputIndex);
        const fromNodeId = connectionByInput.get(key);
        if (fromNodeId != null) {
          const internalIndex = valueIndexByNodeId.get(fromNodeId);
          if (internalIndex == null) throw new Error('사용자 블록 내부 연결을 찾지 못했습니다.');
          inputRefs[inputIndex] = internalIndex;
        } else {
          const externalIndex = externalIndexBySlot.get(key);
          if (externalIndex == null) throw new Error(`사용자 블록 입력 '${blockDef.inputs[inputIndex]}'이 비어 있습니다.`);
          // Negative values encode external input slots: -1 -> slot 0.
          inputRefs[inputIndex] = -externalIndex - 1;
        }
      }

      steps[valueIndex] = {
        valueIndex,
        node,
        kind: 'compute',
        blockDef,
        inputRefs,
        inputs: new Array(inputRefs.length)
      };
    }

    const outputValueIndex = valueIndexByNodeId.get(definition.outputNodeId);
    if (outputValueIndex == null) throw new Error('사용자 블록 출력 노드를 찾지 못했습니다.');

    return {
      definition,
      steps,
      outputValueIndex,
      externalCount: externalInputs.length,
      scratchValues: new Array(steps.length),
      tape: [],
      inUseFrames: [],
      framePool: []
    };
  }

  function planFor(definition) {
    let plan = PLAN_CACHE.get(definition);
    if (!plan) {
      plan = compileUserBlockPlan(definition);
      PLAN_CACHE.set(definition, plan);
    }
    return plan;
  }

  function acquireFrame(plan) {
    const frame = plan.framePool.pop() || {
      values: new Array(plan.steps.length),
      adjoints: new Array(plan.steps.length),
      touchedAdjoints: [],
      externalGrads: new Array(plan.externalCount)
    };
    plan.inUseFrames.push(frame);
    touchedPlans.add(plan);
    return frame;
  }

  function readRef(ref, values, externalValues) {
    return ref >= 0 ? values[ref] : externalValues[-ref - 1];
  }

  function executeUserBlockForward(plan, externalValues, values) {
    const steps = plan.steps;

    for (let s = 0; s < steps.length; s++) {
      const step = steps[s];
      if (step.kind === 'externalSource') {
        values[step.valueIndex] = externalValues[step.sourceSlot];
        continue;
      }

      const inputs = step.inputs;
      for (let i = 0; i < step.inputRefs.length; i++) {
        inputs[i] = readRef(step.inputRefs[i], values, externalValues);
      }
      values[step.valueIndex] = step.blockDef.compute(step.node, inputs);
    }

    return values[plan.outputValueIndex];
  }

  function addGradient(previous, gradient) {
    return previous == null ? gradient : addValues(previous, gradient);
  }

  function executeUserBlockBackward(plan, externalValues, upstream, frame) {
    const values = frame.values;
    const adjoints = frame.adjoints;
    const touched = frame.touchedAdjoints;
    const externalGrads = frame.externalGrads;

    for (let i = 0; i < touched.length; i++) adjoints[touched[i]] = undefined;
    touched.length = 0;
    externalGrads.fill(null);

    adjoints[plan.outputValueIndex] = upstream;
    touched.push(plan.outputValueIndex);

    const steps = plan.steps;
    for (let s = steps.length - 1; s >= 0; s--) {
      const step = steps[s];
      const up = adjoints[step.valueIndex];
      if (up == null) continue;

      if (step.kind === 'externalSource') {
        externalGrads[step.sourceSlot] = addGradient(externalGrads[step.sourceSlot], up);
        continue;
      }

      if (!step.inputRefs.length) continue;

      const inputs = step.inputs;
      for (let i = 0; i < step.inputRefs.length; i++) {
        inputs[i] = readRef(step.inputRefs[i], values, externalValues);
      }

      const grads = step.node.type.startsWith('custom:')
        ? userBlockVJP(USER_BLOCKS.get(step.node.type.slice(7)), inputs, up)
        : primitiveVJP(step.node.type, inputs, values[step.valueIndex], up);

      for (let i = 0; i < grads.length; i++) {
        const gradient = grads[i];
        if (gradient == null) continue;

        const ref = step.inputRefs[i];
        if (ref >= 0) {
          if (adjoints[ref] == null) {
            adjoints[ref] = gradient;
            touched.push(ref);
          } else {
            adjoints[ref] = addValues(adjoints[ref], gradient);
          }
        } else {
          const externalIndex = -ref - 1;
          externalGrads[externalIndex] = addGradient(externalGrads[externalIndex], gradient);
        }
      }
    }

    return externalGrads;
  }

  function clearPlanAfterGradient(plan) {
    plan.tape.length = 0;

    for (const frame of plan.inUseFrames) {
      frame.values.fill(undefined);
      frame.adjoints.fill(undefined);
      frame.touchedAdjoints.length = 0;
      frame.externalGrads.fill(null);
      if (plan.framePool.length < MAX_FRAME_POOL) plan.framePool.push(frame);
    }
    plan.inUseFrames.length = 0;

    // Input arrays are permanent plan objects. Clear stale references so a plan
    // never keeps a large temporary tensor alive after the gradient pass.
    for (const step of plan.steps) {
      if (step.inputs?.length) step.inputs.fill(undefined);
    }
  }

  function finishCapture() {
    for (const plan of touchedPlans) clearPlanAfterGradient(plan);
    touchedPlans = new Set();
  }

  // Replace the recursive/Map-based forward evaluator.
  evaluateUserDefinition = function(definition, externalValues) {
    const plan = planFor(definition);

    if (captureDepth > 0) {
      const frame = acquireFrame(plan);
      const value = executeUserBlockForward(plan, externalValues, frame.values);
      // Each definition owns a LIFO tape. The outer reverse pass visits custom
      // block instances in the exact opposite order of their forward execution.
      plan.tape.push(frame);
      return value;
    }

    return executeUserBlockForward(plan, externalValues, plan.scratchValues);
  };

  // Replace the VJP evaluator. In normal training this consumes the frame made
  // by evaluateUserDefinition above, eliminating the old second internal forward.
  userBlockVJP = function(definition, externalValues, upstream) {
    const plan = planFor(definition);

    if (captureDepth > 0 && plan.tape.length) {
      const frame = plan.tape.pop();
      return executeUserBlockBackward(plan, externalValues, upstream, frame);
    }

    // Safe fallback for a direct VJP call that did not run under the graph-level
    // gradient capture wrapper. It is still compiled; it just needs one forward.
    const frame = {
      values: new Array(plan.steps.length),
      adjoints: new Array(plan.steps.length),
      touchedAdjoints: [],
      externalGrads: new Array(plan.externalCount)
    };
    executeUserBlockForward(plan, externalValues, frame.values);
    return executeUserBlockBackward(plan, externalValues, upstream, frame);
  };

  // app-training-fast replaces computeAllGraphGradients before this file loads.
  // Wrap that final fast implementation so every custom forward produced inside
  // a gradient pass has a matching preserved frame for reverse mode.
  const computeAllGraphGradientsBeforeCompiledUserBlocks = computeAllGraphGradients;
  computeAllGraphGradients = function(outputId) {
    const outermost = captureDepth === 0;
    if (outermost) {
      finishCapture();
      touchedPlans = new Set();
    }

    captureDepth++;
    try {
      return computeAllGraphGradientsBeforeCompiledUserBlocks(outputId);
    } finally {
      captureDepth--;
      if (outermost) finishCapture();
    }
  };
})();
