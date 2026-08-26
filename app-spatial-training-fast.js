// Training-only fusion for sliding-window unfold -> matrix x vector.
//
// The visible graph and math stay unchanged. For reverse-mode training, an
// unfold2d whose consumers are all matvec blocks can be executed without ever
// materializing its large patch matrix or the equally large patch gradients.
//
// Floating-point order is intentionally preserved. If one unfold feeds several
// matvec nodes (for example four convolution filters), the normal engine first
// accumulates their Float32 patch gradients and only then scatters them back to
// the source image. The fused path reproduces that order.

(function installSpatialTrainingFusion() {
  const computeAllGraphGradientsBeforeSpatialFusion = computeAllGraphGradients;
  let planVersion = '';
  const plans = new Map();
  let lastFusionCount = 0;

  function positiveInt(value, fallback = 1) {
    const n = Math.floor(Number(value));
    return Number.isFinite(n) && n > 0 ? n : fallback;
  }

  function nonNegativeInt(value, fallback = 0) {
    const n = Math.floor(Number(value));
    return Number.isFinite(n) && n >= 0 ? n : fallback;
  }

  function unfoldConfig(node, input) {
    const arr = asArrayValue(input);
    if (arr.shape.length !== 2) throw new Error('슬라이딩 창 펼치기에는 2차원 배열이 필요합니다.');

    const height = arr.shape[0];
    const width = arr.shape[1];
    const kernelRows = positiveInt(node.params.kernelRows, 3);
    const kernelCols = positiveInt(node.params.kernelCols, 3);
    const strideRows = positiveInt(node.params.strideRows, 1);
    const strideCols = positiveInt(node.params.strideCols, 1);
    const padding = nonNegativeInt(node.params.padding, 0);
    const paddedHeight = height + padding * 2;
    const paddedWidth = width + padding * 2;

    if (kernelRows > paddedHeight || kernelCols > paddedWidth) {
      throw new Error('슬라이딩 창이 패딩을 포함한 입력보다 큽니다.');
    }

    const outRows = Math.floor((paddedHeight - kernelRows) / strideRows) + 1;
    const outCols = Math.floor((paddedWidth - kernelCols) / strideCols) + 1;
    if (outRows < 1 || outCols < 1) throw new Error('슬라이딩 창의 출력 크기가 0입니다.');

    return {
      height,
      width,
      kernelRows,
      kernelCols,
      strideRows,
      strideCols,
      padding,
      outRows,
      outCols,
      windowSize: kernelRows * kernelCols,
      windowCount: outRows * outCols
    };
  }

  function validateVector(config, vector) {
    const v = asArrayValue(vector);
    if (v.shape.length !== 1) throw new Error('두 번째 입력은 벡터여야 합니다.');
    if (v.data.length !== config.windowSize) {
      throw new Error(`행렬의 열 ${config.windowSize}개와 벡터 길이 ${v.data.length}가 다릅니다.`);
    }
    return v;
  }

  // ---------- fused forward ----------

  function forward3x3Stride1(source, vector, config) {
    const src = source.data;
    const vd = vector.data;
    const out = takeResultBuffer(config.windowCount);
    const width = config.width;
    let write = 0;

    for (let r = 0; r < config.outRows; r++) {
      let p = r * width;
      for (let c = 0; c < config.outCols; c++, p++) {
        let sum = 0;
        sum += src[p] * vd[0];
        sum += src[p + 1] * vd[1];
        sum += src[p + 2] * vd[2];
        sum += src[p + width] * vd[3];
        sum += src[p + width + 1] * vd[4];
        sum += src[p + width + 2] * vd[5];
        sum += src[p + width * 2] * vd[6];
        sum += src[p + width * 2 + 1] * vd[7];
        sum += src[p + width * 2 + 2] * vd[8];
        out[write++] = sum;
      }
    }
    return fastArrayValue(out, [config.windowCount]);
  }

  function forward2x2Stride2(source, vector, config) {
    const src = source.data;
    const vd = vector.data;
    const out = takeResultBuffer(config.windowCount);
    const width = config.width;
    let write = 0;

    for (let r = 0; r < config.outRows; r++) {
      let p = r * 2 * width;
      for (let c = 0; c < config.outCols; c++, p += 2) {
        let sum = 0;
        sum += src[p] * vd[0];
        sum += src[p + 1] * vd[1];
        sum += src[p + width] * vd[2];
        sum += src[p + width + 1] * vd[3];
        out[write++] = sum;
      }
    }
    return fastArrayValue(out, [config.windowCount]);
  }

  function forwardGeneric(source, vector, config) {
    const src = source.data;
    const vd = vector.data;
    const out = takeResultBuffer(config.windowCount);
    let write = 0;

    for (let outRow = 0; outRow < config.outRows; outRow++) {
      const sourceTop = outRow * config.strideRows - config.padding;
      for (let outCol = 0; outCol < config.outCols; outCol++) {
        const sourceLeft = outCol * config.strideCols - config.padding;
        let sum = 0;
        let k = 0;
        for (let kr = 0; kr < config.kernelRows; kr++) {
          const sourceRow = sourceTop + kr;
          for (let kc = 0; kc < config.kernelCols; kc++, k++) {
            const sourceCol = sourceLeft + kc;
            const value = sourceRow >= 0 && sourceRow < config.height && sourceCol >= 0 && sourceCol < config.width
              ? src[sourceRow * config.width + sourceCol]
              : 0;
            sum += value * vd[k];
          }
        }
        out[write++] = sum;
      }
    }
    return fastArrayValue(out, [config.windowCount]);
  }

  function unfoldMatvecForward(unfoldNode, input, vector) {
    const source = asArrayValue(input);
    const config = unfoldConfig(unfoldNode, source);
    const v = validateVector(config, vector);

    if (config.padding === 0 && config.kernelRows === 3 && config.kernelCols === 3 &&
        config.strideRows === 1 && config.strideCols === 1) {
      return forward3x3Stride1(source, v, config);
    }
    if (config.padding === 0 && config.kernelRows === 2 && config.kernelCols === 2 &&
        config.strideRows === 2 && config.strideCols === 2) {
      return forward2x2Stride2(source, v, config);
    }
    return forwardGeneric(source, v, config);
  }

  // ---------- fused backward: matvec vector input ----------

  function vectorGradient3x3Stride1(source, upstream, config) {
    const src = source.data;
    const ud = upstream.data;
    const width = config.width;
    let g0 = 0, g1 = 0, g2 = 0, g3 = 0, g4 = 0, g5 = 0, g6 = 0, g7 = 0, g8 = 0;
    let read = 0;

    for (let r = 0; r < config.outRows; r++) {
      let p = r * width;
      for (let c = 0; c < config.outCols; c++, p++) {
        const u = ud[read++];
        g0 += src[p] * u;
        g1 += src[p + 1] * u;
        g2 += src[p + 2] * u;
        g3 += src[p + width] * u;
        g4 += src[p + width + 1] * u;
        g5 += src[p + width + 2] * u;
        g6 += src[p + width * 2] * u;
        g7 += src[p + width * 2 + 1] * u;
        g8 += src[p + width * 2 + 2] * u;
      }
    }

    const out = takeResultBuffer(9);
    out[0] = g0; out[1] = g1; out[2] = g2;
    out[3] = g3; out[4] = g4; out[5] = g5;
    out[6] = g6; out[7] = g7; out[8] = g8;
    return fastArrayValue(out, [9]);
  }

  function vectorGradient2x2Stride2(source, upstream, config) {
    const src = source.data;
    const ud = upstream.data;
    const width = config.width;
    let g0 = 0, g1 = 0, g2 = 0, g3 = 0;
    let read = 0;

    for (let r = 0; r < config.outRows; r++) {
      let p = r * 2 * width;
      for (let c = 0; c < config.outCols; c++, p += 2) {
        const u = ud[read++];
        g0 += src[p] * u;
        g1 += src[p + 1] * u;
        g2 += src[p + width] * u;
        g3 += src[p + width + 1] * u;
      }
    }

    const out = takeResultBuffer(4);
    out[0] = g0; out[1] = g1; out[2] = g2; out[3] = g3;
    return fastArrayValue(out, [4]);
  }

  function vectorGradientGeneric(source, upstream, config) {
    const src = source.data;
    const ud = upstream.data;
    const sums = new Float64Array(config.windowSize);
    let read = 0;

    for (let outRow = 0; outRow < config.outRows; outRow++) {
      const sourceTop = outRow * config.strideRows - config.padding;
      for (let outCol = 0; outCol < config.outCols; outCol++) {
        const sourceLeft = outCol * config.strideCols - config.padding;
        const u = ud[read++];
        let k = 0;
        for (let kr = 0; kr < config.kernelRows; kr++) {
          const sourceRow = sourceTop + kr;
          for (let kc = 0; kc < config.kernelCols; kc++, k++) {
            const sourceCol = sourceLeft + kc;
            if (sourceRow >= 0 && sourceRow < config.height && sourceCol >= 0 && sourceCol < config.width) {
              sums[k] += src[sourceRow * config.width + sourceCol] * u;
            }
          }
        }
      }
    }

    const out = takeResultBuffer(config.windowSize);
    for (let k = 0; k < config.windowSize; k++) out[k] = sums[k];
    return fastArrayValue(out, [config.windowSize]);
  }

  function unfoldMatvecVectorGradient(unfoldNode, input, upstream) {
    const source = asArrayValue(input);
    const gradient = asArrayValue(upstream);
    const config = unfoldConfig(unfoldNode, source);
    if (gradient.shape.length !== 1 || gradient.data.length !== config.windowCount) {
      throw new Error('행렬 × 벡터 역전파의 gradient 크기가 맞지 않습니다.');
    }

    if (config.padding === 0 && config.kernelRows === 3 && config.kernelCols === 3 &&
        config.strideRows === 1 && config.strideCols === 1) {
      return vectorGradient3x3Stride1(source, gradient, config);
    }
    if (config.padding === 0 && config.kernelRows === 2 && config.kernelCols === 2 &&
        config.strideRows === 2 && config.strideCols === 2) {
      return vectorGradient2x2Stride2(source, gradient, config);
    }
    return vectorGradientGeneric(source, gradient, config);
  }

  // ---------- fused backward: unfold source input ----------

  function contributionArrays(contributionSets) {
    const upstreams = new Array(contributionSets.length);
    const vectors = new Array(contributionSets.length);
    for (let i = 0; i < contributionSets.length; i++) {
      upstreams[i] = contributionSets[i].upstream.data;
      vectors[i] = contributionSets[i].vector.data;
    }
    return { upstreams, vectors };
  }

  function combinedPatchGradient(upstreams, vectors, row, patchIndex) {
    let value = Math.fround(upstreams[0][row] * vectors[0][patchIndex]);
    for (let i = 1; i < upstreams.length; i++) {
      const next = Math.fround(upstreams[i][row] * vectors[i][patchIndex]);
      value = Math.fround(value + next);
    }
    return value;
  }

  function inputGradient3x3Stride1(source, contributionSets, config) {
    const out = takeResultBuffer(source.data.length);
    out.fill(0);
    const width = config.width;
    const { upstreams, vectors } = contributionArrays(contributionSets);

    for (let r = 0, row = 0; r < config.outRows; r++) {
      let p = r * width;
      for (let c = 0; c < config.outCols; c++, p++, row++) {
        out[p] += combinedPatchGradient(upstreams, vectors, row, 0);
        out[p + 1] += combinedPatchGradient(upstreams, vectors, row, 1);
        out[p + 2] += combinedPatchGradient(upstreams, vectors, row, 2);
        out[p + width] += combinedPatchGradient(upstreams, vectors, row, 3);
        out[p + width + 1] += combinedPatchGradient(upstreams, vectors, row, 4);
        out[p + width + 2] += combinedPatchGradient(upstreams, vectors, row, 5);
        out[p + width * 2] += combinedPatchGradient(upstreams, vectors, row, 6);
        out[p + width * 2 + 1] += combinedPatchGradient(upstreams, vectors, row, 7);
        out[p + width * 2 + 2] += combinedPatchGradient(upstreams, vectors, row, 8);
      }
    }
    return fastArrayValue(out, source.shape);
  }

  function inputGradient2x2Stride2(source, contributionSets, config) {
    const out = takeResultBuffer(source.data.length);
    out.fill(0);
    const width = config.width;
    const { upstreams, vectors } = contributionArrays(contributionSets);

    for (let r = 0, row = 0; r < config.outRows; r++) {
      let p = r * 2 * width;
      for (let c = 0; c < config.outCols; c++, p += 2, row++) {
        out[p] += combinedPatchGradient(upstreams, vectors, row, 0);
        out[p + 1] += combinedPatchGradient(upstreams, vectors, row, 1);
        out[p + width] += combinedPatchGradient(upstreams, vectors, row, 2);
        out[p + width + 1] += combinedPatchGradient(upstreams, vectors, row, 3);
      }
    }
    return fastArrayValue(out, source.shape);
  }

  function inputGradientGeneric(source, contributionSets, config) {
    const out = takeResultBuffer(source.data.length);
    out.fill(0);
    const { upstreams, vectors } = contributionArrays(contributionSets);

    for (let outRow = 0, row = 0; outRow < config.outRows; outRow++) {
      const sourceTop = outRow * config.strideRows - config.padding;
      for (let outCol = 0; outCol < config.outCols; outCol++, row++) {
        const sourceLeft = outCol * config.strideCols - config.padding;
        let k = 0;
        for (let kr = 0; kr < config.kernelRows; kr++) {
          const sourceRow = sourceTop + kr;
          for (let kc = 0; kc < config.kernelCols; kc++, k++) {
            const sourceCol = sourceLeft + kc;
            if (sourceRow < 0 || sourceRow >= config.height || sourceCol < 0 || sourceCol >= config.width) continue;
            out[sourceRow * config.width + sourceCol] += combinedPatchGradient(upstreams, vectors, row, k);
          }
        }
      }
    }
    return fastArrayValue(out, source.shape);
  }

  function unfoldInputGradient(unfoldNode, input, contributionSets) {
    const source = asArrayValue(input);
    const config = unfoldConfig(unfoldNode, source);
    if (!contributionSets.length) return zerosLike(source);

    for (const contribution of contributionSets) {
      validateVector(config, contribution.vector);
      const upstream = asArrayValue(contribution.upstream);
      if (upstream.shape.length !== 1 || upstream.data.length !== config.windowCount) {
        throw new Error('행렬 × 벡터 역전파의 gradient 크기가 맞지 않습니다.');
      }
    }

    if (config.padding === 0 && config.kernelRows === 3 && config.kernelCols === 3 &&
        config.strideRows === 1 && config.strideCols === 1) {
      return inputGradient3x3Stride1(source, contributionSets, config);
    }
    if (config.padding === 0 && config.kernelRows === 2 && config.kernelCols === 2 &&
        config.strideRows === 2 && config.strideCols === 2) {
      return inputGradient2x2Stride2(source, contributionSets, config);
    }
    return inputGradientGeneric(source, contributionSets, config);
  }

  // ---------- compiled reverse-mode plan ----------

  function structureVersion(cache) {
    return `${cache.nodeCount}:${cache.connectionCount}:${cache.hashA}:${cache.hashB}`;
  }

  function compilePlan(outputId, structureCache) {
    const topo = cachedTopoForOutput(structureCache, outputId);
    const steps = [];
    const variableSteps = [];

    for (const id of topo) {
      const node = graph.nodes.get(id);
      if (!node) throw new Error('존재하지 않는 블록입니다.');
      const def = getBlockDef(node.type);
      if (def.special === 'repeat' || def.special === 'setVariable' || def.special === 'derivative') return null;

      const inputIds = new Array(def.inputs.length);
      for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
        const connection = cachedGraphInput(structureCache, id, inputIndex);
        if (!connection) throw new Error(`입력 '${def.inputs[inputIndex]}'이 연결되지 않았습니다.`);
        inputIds[inputIndex] = connection.from;
      }

      const step = {
        id,
        node,
        def,
        inputIds,
        inputs: new Array(inputIds.length),
        kind: 'normal',
        unfoldId: null,
        unfoldNode: null,
        unfoldInputId: null,
        vectorInputId: null
      };
      steps.push(step);
      if (node.type === 'variable') variableSteps.push(step);
    }

    const consumers = new Map();
    for (const step of steps) {
      for (const inputId of step.inputIds) {
        let list = consumers.get(inputId);
        if (!list) {
          list = [];
          consumers.set(inputId, list);
        }
        list.push(step);
      }
    }

    let fusionCount = 0;
    for (const step of steps) {
      if (step.node.type !== 'unfold2d' || step.inputIds.length !== 1) continue;
      const list = consumers.get(step.id) || [];
      if (!list.length) continue;
      if (!list.every(consumer => consumer.node.type === 'matvec' && consumer.inputIds[0] === step.id)) continue;

      step.kind = 'fusedUnfoldSource';
      for (const consumer of list) {
        consumer.kind = 'fusedUnfoldMatvec';
        consumer.unfoldId = step.id;
        consumer.unfoldNode = step.node;
        consumer.unfoldInputId = step.inputIds[0];
        consumer.vectorInputId = consumer.inputIds[1];
        fusionCount++;
      }
    }

    if (!fusionCount) return null;
    return {
      outputId,
      steps,
      variableSteps,
      values: [],
      adjoints: [],
      touchedAdjoints: [],
      pendingUnfoldAdjoints: new Map(),
      fusionCount
    };
  }

  function getPlan(outputId, structureCache) {
    const version = structureVersion(structureCache);
    if (planVersion !== version) {
      planVersion = version;
      plans.clear();
    }
    if (plans.has(outputId)) return plans.get(outputId);
    const plan = compilePlan(outputId, structureCache);
    plans.set(outputId, plan);
    return plan;
  }

  function accumulateAdjoint(plan, inputId, gradient) {
    const previous = plan.adjoints[inputId];
    if (previous == null) {
      plan.adjoints[inputId] = gradient;
      plan.touchedAdjoints.push(inputId);
    } else {
      plan.adjoints[inputId] = addValues(previous, gradient);
    }
  }

  function executePlan(plan) {
    const values = plan.values;
    const adjoints = plan.adjoints;
    const touched = plan.touchedAdjoints;
    const pendingUnfoldAdjoints = plan.pendingUnfoldAdjoints;

    for (let i = 0; i < touched.length; i++) adjoints[touched[i]] = undefined;
    touched.length = 0;
    pendingUnfoldAdjoints.clear();

    for (const step of plan.steps) {
      if (step.kind === 'fusedUnfoldSource') {
        values[step.id] = undefined;
        continue;
      }
      if (step.kind === 'fusedUnfoldMatvec') {
        values[step.id] = unfoldMatvecForward(step.unfoldNode, values[step.unfoldInputId], values[step.vectorInputId]);
        continue;
      }

      const inputs = step.inputs;
      for (let i = 0; i < step.inputIds.length; i++) inputs[i] = values[step.inputIds[i]];
      values[step.id] = step.def.compute(step.node, inputs);
    }

    const output = values[plan.outputId];
    if (typeof output !== 'number') throw new Error('미분 블록의 식 출력은 숫자 하나여야 합니다.');
    adjoints[plan.outputId] = 1;
    touched.push(plan.outputId);

    for (let k = plan.steps.length - 1; k >= 0; k--) {
      const step = plan.steps[k];
      const upstream = adjoints[step.id];

      if (step.kind === 'fusedUnfoldMatvec') {
        if (upstream == null) continue;
        const gradient = asArrayValue(upstream);
        const source = values[step.unfoldInputId];
        const vector = values[step.vectorInputId];

        accumulateAdjoint(plan, step.vectorInputId, unfoldMatvecVectorGradient(step.unfoldNode, source, gradient));

        let contributions = pendingUnfoldAdjoints.get(step.unfoldId);
        if (!contributions) {
          contributions = [];
          pendingUnfoldAdjoints.set(step.unfoldId, contributions);
        }
        contributions.push({ upstream: gradient, vector: asArrayValue(vector) });
        continue;
      }

      if (step.kind === 'fusedUnfoldSource') {
        const contributions = pendingUnfoldAdjoints.get(step.id);
        if (!contributions?.length) continue;
        accumulateAdjoint(plan, step.inputIds[0], unfoldInputGradient(step.node, values[step.inputIds[0]], contributions));
        continue;
      }

      if (upstream == null || !step.inputIds.length) continue;
      const inputs = step.inputs;
      const grads = step.node.type.startsWith('custom:')
        ? userBlockVJP(USER_BLOCKS.get(step.node.type.slice(7)), inputs, upstream)
        : primitiveVJP(step.node.type, inputs, values[step.id], upstream);

      for (let i = 0; i < grads.length; i++) {
        if (grads[i] != null) accumulateAdjoint(plan, step.inputIds[i], grads[i]);
      }
    }

    const gradients = new Map();
    for (const step of plan.variableSteps) {
      const name = String(step.node.params.name || 'x');
      const gradient = adjoints[step.id] ?? zerosLike(values[step.id]);
      const previous = gradients.get(name);
      gradients.set(name, previous == null ? gradient : addValues(previous, gradient));
    }
    lastFusionCount = plan.fusionCount;
    return gradients;
  }

  computeAllGraphGradients = function(outputId) {
    if (window.SPATIAL_FUSION_DISABLED === true) {
      lastFusionCount = 0;
      return computeAllGraphGradientsBeforeSpatialFusion(outputId);
    }

    const structureCache = ensureGraphStructureCache();
    const plan = getPlan(outputId, structureCache);
    if (!plan) {
      lastFusionCount = 0;
      return computeAllGraphGradientsBeforeSpatialFusion(outputId);
    }
    return executePlan(plan);
  };

  window.SPATIAL_TRAINING_FUSION = {
    get lastFusionCount() { return lastFusionCount; },
    setDisabled(value) { window.SPATIAL_FUSION_DISABLED = Boolean(value); },
    clearCache() {
      planVersion = '';
      plans.clear();
    }
  };
})();
