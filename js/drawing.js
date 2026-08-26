// The drawing pad, Quick, Draw!-compatible preprocessing, and the hand-drawing
// test button.
//
// The training bitmaps were generated from Quick Draw's *simplified vector*
// representation, not by shrinking an already-rasterized browser canvas. So the
// pad records pointer strokes as vectors, applies the documented Quick Draw
// simplification pipeline, and hands those vectors to the real Cairo image
// backend compiled to WebAssembly. The visible brush width is cosmetic; the
// 28×28 model input never comes from the canvas pixels.
//
// Simplification documented by googlecreativelab/quickdraw-dataset:
//   1. top-left align (minimum x/y = 0)
//   2. uniformly scale so the largest extent is 255
//   3. resample each stroke at 1-pixel spacing
//   4. Ramer-Douglas-Peucker simplify with epsilon = 2.0
//
// The Cairo wrapper then reproduces the published 28x28 rasterizer:
// ANTIALIAS_BEST, ROUND caps/joins, line diameter 16, padding 16 and centered
// bounding box. No browser canvas resize or gamma correction is used.

const OUTPUT_SIZE = 28;
const SOURCE_SIDE = 256;
const RESAMPLE_SPACING = 1;
const RDP_EPSILON = 2;
const DISPLAY_LINE_WIDTH = 10;
const CAIRO_WASM_BASE = 'vendor/quickdraw-cairo';
const CAIRO_ASSET_VERSION = '20260826-1';

const rawStrokes = [];
let activeStroke = null;
let drawingPointerDown = false;
let lastDrawPoint = { x: 0, y: 0 };
let cairoModule = null;
let cairoError = null;
let cairoVersion = null;

function pointerOnDraw(event) {
  const rect = drawCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * drawCanvas.width / rect.width,
    y: (event.clientY - rect.top) * drawCanvas.height / rect.height
  };
}

// ---------- the visible pad, and the vector recording behind it ----------

function appendStrokePoint(stroke, point) {
  if (!stroke) return;
  const x = Number(point.x);
  const y = Number(point.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return;
  const last = stroke[stroke.length - 1];
  if (last && last.x === x && last.y === y) return;
  stroke.push({ x, y });
}

function pointerSamples(event) {
  const samples = typeof event.getCoalescedEvents === 'function' ? event.getCoalescedEvents() : null;
  return samples && samples.length ? samples : [event];
}

function resetVectorDrawing() {
  rawStrokes.length = 0;
  activeStroke = null;
}

function resetPreview() {
  previewCtx.fillStyle = '#000';
  previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
}

function resetDrawCanvas() {
  resetVectorDrawing();
  drawCtx.fillStyle = '#fff';
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  drawCtx.lineCap = 'round';
  drawCtx.lineJoin = 'round';
  drawCtx.lineWidth = DISPLAY_LINE_WIDTH;
  drawCtx.strokeStyle = '#111';
  resetPreview();
}

function renderPreview(data) {
  const tiny = document.createElement('canvas');
  tiny.width = 28;
  tiny.height = 28;
  const tctx = tiny.getContext('2d'), image = tctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = Math.round(clamp(data[i], 0, 1) * 255), p = i * 4;
    image.data[p] = v; image.data[p + 1] = v; image.data[p + 2] = v; image.data[p + 3] = 255;
  }
  tctx.putImageData(image, 0, 0);
  previewCtx.imageSmoothingEnabled = false;
  previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  previewCtx.drawImage(tiny, 0, 0, previewCanvas.width, previewCanvas.height);
}

drawCanvas.addEventListener('pointerdown', event => {
  drawingPointerDown = true;
  lastDrawPoint = pointerOnDraw(event);
  drawCanvas.setPointerCapture?.(event.pointerId);

  activeStroke = [];
  rawStrokes.push(activeStroke);
  appendStrokePoint(activeStroke, lastDrawPoint);
});

drawCanvas.addEventListener('pointermove', event => {
  if (!drawingPointerDown) return;
  const p = pointerOnDraw(event);
  drawCtx.beginPath();
  drawCtx.moveTo(lastDrawPoint.x, lastDrawPoint.y);
  drawCtx.lineTo(p.x, p.y);
  drawCtx.stroke();
  lastDrawPoint = p;

  for (const sample of pointerSamples(event)) appendStrokePoint(activeStroke, pointerOnDraw(sample));
});

function finishDrawStroke(event) {
  drawingPointerDown = false;
  if (event?.pointerId != null && drawCanvas.hasPointerCapture?.(event.pointerId)) {
    drawCanvas.releasePointerCapture(event.pointerId);
  }
  if (activeStroke) {
    if (event) appendStrokePoint(activeStroke, pointerOnDraw(event));
    if (!activeStroke.length) rawStrokes.pop();
    activeStroke = null;
  }
  invalidatePreviews();
}

drawCanvas.addEventListener('pointerup', finishDrawStroke);
drawCanvas.addEventListener('pointercancel', finishDrawStroke);

// ---------- Quick Draw stroke simplification ----------

function strokeDistance(a, b) {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function topLeftScale(strokes) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const stroke of strokes) {
    for (const p of stroke) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
  }

  if (!Number.isFinite(minX)) return [];
  const width = maxX - minX;
  const height = maxY - minY;
  const largestExtent = Math.max(width, height);
  const scale = largestExtent > 0 ? (SOURCE_SIDE - 1) / largestExtent : 1;

  return strokes.map(stroke => stroke.map(p => ({
    x: (p.x - minX) * scale,
    y: (p.y - minY) * scale
  })));
}

function resampleStroke(points, spacing = RESAMPLE_SPACING) {
  if (points.length <= 1) return points.map(p => ({ ...p }));

  const output = [{ ...points[0] }];
  let segmentStart = { ...points[0] };
  let distanceFromLastSample = 0;

  for (let i = 1; i < points.length; i++) {
    const segmentEnd = points[i];
    let segmentLength = strokeDistance(segmentStart, segmentEnd);
    if (segmentLength <= 1e-12) {
      segmentStart = { ...segmentEnd };
      continue;
    }

    while (distanceFromLastSample + segmentLength >= spacing) {
      const needed = spacing - distanceFromLastSample;
      const t = needed / segmentLength;
      const sample = {
        x: segmentStart.x + (segmentEnd.x - segmentStart.x) * t,
        y: segmentStart.y + (segmentEnd.y - segmentStart.y) * t
      };
      output.push(sample);
      segmentStart = sample;
      segmentLength = strokeDistance(segmentStart, segmentEnd);
      distanceFromLastSample = 0;
      if (segmentLength <= 1e-12) break;
    }

    distanceFromLastSample += segmentLength;
    segmentStart = { ...segmentEnd };
  }

  const finalPoint = points[points.length - 1];
  if (strokeDistance(output[output.length - 1], finalPoint) > 1e-9) output.push({ ...finalPoint });
  return output;
}

function pointSegmentDistance(point, start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared <= 1e-20) return strokeDistance(point, start);
  const t = Math.max(0, Math.min(1,
    ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared
  ));
  const projection = { x: start.x + t * dx, y: start.y + t * dy };
  return strokeDistance(point, projection);
}

function rdpStroke(points, epsilon = RDP_EPSILON) {
  if (points.length <= 2) return points.map(p => ({ ...p }));

  const keep = new Uint8Array(points.length);
  keep[0] = 1;
  keep[points.length - 1] = 1;
  const stack = [[0, points.length - 1]];

  while (stack.length) {
    const [startIndex, endIndex] = stack.pop();
    const start = points[startIndex];
    const end = points[endIndex];
    let farthestIndex = -1;
    let farthestDistance = epsilon;

    for (let i = startIndex + 1; i < endIndex; i++) {
      const d = pointSegmentDistance(points[i], start, end);
      if (d > farthestDistance) {
        farthestDistance = d;
        farthestIndex = i;
      }
    }

    if (farthestIndex >= 0) {
      keep[farthestIndex] = 1;
      stack.push([startIndex, farthestIndex], [farthestIndex, endIndex]);
    }
  }

  const simplified = [];
  for (let i = 0; i < points.length; i++) if (keep[i]) simplified.push({ ...points[i] });
  return simplified;
}

function simplifyCurrentDrawing() {
  const nonEmpty = rawStrokes.filter(stroke => stroke.length).map(stroke => stroke.map(p => ({ ...p })));
  if (!nonEmpty.length) return [];
  const scaled = topLeftScale(nonEmpty);
  return scaled.map(stroke => rdpStroke(resampleStroke(stroke)));
}

// ---------- Cairo rasterization ----------

function cairoNotReadyMessage() {
  if (cairoError) return `Quick Draw Cairo WASM 준비 실패: ${cairoError.message || cairoError}`;
  return 'Quick Draw Cairo WASM을 불러오는 중입니다. 잠시 뒤 다시 시도하세요.';
}

function renderWithCairo(strokes) {
  if (!cairoModule) throw new Error(cairoNotReadyMessage());
  if (!strokes.length) return arrayValue(new Float32Array(OUTPUT_SIZE * OUTPUT_SIZE), [OUTPUT_SIZE, OUTPUT_SIZE]);

  cairoModule._qd_reset();
  for (const stroke of strokes) {
    if (!stroke.length) continue;
    if (cairoModule._qd_begin_stroke() < 0) throw new Error('Cairo 입력 stroke 수가 제한을 초과했습니다.');
    for (const p of stroke) {
      const status = cairoModule._qd_add_point(p.x, p.y);
      if (status !== 0) throw new Error(`Cairo 입력 point 수가 제한을 초과했습니다 (${status}).`);
    }
  }

  const status = cairoModule._qd_render();
  if (status !== 0) throw new Error(`Cairo 28×28 렌더링 실패 (${status}).`);
  const ptr = Number(cairoModule._qd_raster_ptr());
  const length = Number(cairoModule._qd_raster_length());
  if (length !== OUTPUT_SIZE * OUTPUT_SIZE) throw new Error(`Cairo 출력 길이가 잘못되었습니다: ${length}`);

  const source = cairoModule.HEAPU8.subarray(ptr, ptr + length);
  const data = new Float32Array(length);
  for (let i = 0; i < length; i++) data[i] = source[i] / 255;
  return arrayValue(data, [OUTPUT_SIZE, OUTPUT_SIZE]);
}

// The canvas argument is part of the block interface but unused: the model input
// comes from the recorded vector path, not from canvas pixels.
function preprocessCanvasTo28() {
  return renderWithCairo(simplifyCurrentDrawing());
}

function loadCairoRuntimeScript() {
  if (typeof QuickDrawCairoModule === 'function') return Promise.resolve();
  return new Promise((resolve, reject) => {
    const existing = document.querySelector('script[data-quickdraw-cairo-runtime]');
    if (existing) {
      existing.addEventListener('load', resolve, { once: true });
      existing.addEventListener('error', () => reject(new Error('Cairo runtime script 로드 실패')), { once: true });
      return;
    }
    const script = document.createElement('script');
    script.dataset.quickdrawCairoRuntime = '1';
    script.src = `${CAIRO_WASM_BASE}.js?v=${CAIRO_ASSET_VERSION}`;
    script.async = true;
    script.addEventListener('load', resolve, { once: true });
    script.addEventListener('error', () => reject(new Error(`${script.src} 로드 실패`)), { once: true });
    document.head.appendChild(script);
  });
}

async function initializeCairo() {
  try {
    await loadCairoRuntimeScript();
    if (typeof QuickDrawCairoModule !== 'function') throw new Error('QuickDrawCairoModule 함수가 생성되지 않았습니다.');
    cairoModule = await QuickDrawCairoModule({
      locateFile(path) {
        if (path.endsWith('.wasm')) return `${CAIRO_WASM_BASE}.wasm?v=${CAIRO_ASSET_VERSION}`;
        return path;
      }
    });
    const versionPtr = Number(cairoModule._qd_cairo_version());
    if (versionPtr && typeof cairoModule.UTF8ToString === 'function') cairoVersion = cairoModule.UTF8ToString(versionPtr);
    document.documentElement.dataset.quickDrawCairo = 'ready';
    return cairoModule;
  } catch (error) {
    cairoError = error;
    document.documentElement.dataset.quickDrawCairo = 'error';
    console.error('Quick Draw Cairo WASM initialization failed', error);
    throw error;
  }
}

const quickDrawCairoReady = initializeCairo();
// Prevent an unhandled-rejection warning while still exposing the promise to
// tests/developer tools. User-facing compute paths report the stored error.
quickDrawCairoReady.catch(() => {});

window.QUICKDRAW_CAIRO = {
  ready: quickDrawCairoReady,
  get module() { return cairoModule; },
  get error() { return cairoError; },
  get version() { return cairoVersion; },
  get rawStrokeCount() { return rawStrokes.length; },
  simplifyCurrentDrawing,
  reset: resetVectorDrawing
};

// ---------- 손그림 테스트 ----------

function formatClassifierScores(classes, scores) {
  return classes.map((name, i) => `${koreanClassName(name)} ${Number(scores.data[i]).toFixed(3)}`).join(' · ');
}

function setDrawingClassifierResult(message, isError = false) {
  const result = document.getElementById('drawingClassificationResult');
  if (!result) return;
  result.textContent = message;
  result.classList.toggle('classification-error', isError);
}

function classifyCurrentDrawing() {
  try {
    if (selectedNodeId == null) throw new Error('먼저 작업공간에서 모델의 최종 점수 출력 블록을 선택하세요.');

    const classes = window.quickDrawDataset?.getLoadedClassNames?.() || [];
    if (!classes.length) throw new Error('먼저 Quick Draw 종류를 불러오세요.');

    const ancestors = collectAncestorNodeIds(selectedNodeId);
    let hasSampleImage = false;
    for (const id of ancestors) {
      const node = graph.nodes.get(id);
      if (!node) continue;
      const def = getBlockDef(node.type);
      if (node.type === 'sampleImage') hasSampleImage = true;
      if (def.special === 'repeat' || def.special === 'setVariable' || def.special === 'derivative') {
        throw new Error('학습·값 변경·미분 블록이 포함된 출력입니다. 순수한 예측 점수 출력 블록을 선택하세요.');
      }
    }
    if (!hasSampleImage) {
      throw new Error("선택한 출력이 '그림값' 블록을 사용하지 않습니다. 학습된 모델의 점수 출력 블록을 선택하세요.");
    }

    const drawing = preprocessCanvasTo28(drawCanvas);
    const inkAmount = sumArray(drawing);
    if (inkAmount <= 0.001) throw new Error('그림판에 먼저 그림을 그리세요.');
    renderPreview(drawing.data);

    // 그림값 reads the hand drawing for the duration of this one evaluation.
    setHandDrawingSampleOverride(() => arrayValue(new Float32Array(drawing.data), [28, 28]));
    let scores;
    try {
      scores = evaluateNode(selectedNodeId, new Map(), new Set());
    } finally {
      setHandDrawingSampleOverride(null);
    }

    const vector = asArrayValue(scores);
    if (vector.shape.length !== 1) throw new Error('선택한 출력이 점수 벡터가 아닙니다.');
    if (vector.data.length !== classes.length) {
      throw new Error(`출력 길이 ${vector.data.length}와 현재 데이터 종류 수 ${classes.length}가 다릅니다.`);
    }

    const index = argmaxValue(vector);
    const className = classes[index];
    setDrawingClassifierResult(`${koreanClassName(className)} (${className}) · ${formatClassifierScores(classes, vector)}`);

    const selected = graph.nodes.get(selectedNodeId);
    if (selected) {
      selected.lastValue = copyValue(vector);
      selected.lastError = null;
      updateNodePreview(selected);
      renderInspector();
    }
  } catch (error) {
    setDrawingClassifierResult(error.message, true);
  }
}

document.getElementById('classifyDrawingBtn')?.addEventListener('click', classifyCurrentDrawing);
