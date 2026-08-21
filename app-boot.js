(function installProgressRateDisplay() {
  // readRuntimeVariable, writeRuntimeVariable, accumulateGrad and
  // differentiateGraph used to be re-declared here without their defensive
  // copies. Those copies are now gone from the original definitions, so the
  // overrides were removed: the code you read in app-core.js, app-training.js
  // and app-engine.js is the code that runs.
  function updateStepRate(progress, now, force = false) {
    if (progress.rateSampleTime == null) {
      progress.rateSampleTime = now;
      progress.rateSampleCompleted = progress.completed;
      progress.stepsPerSecond = null;
      return;
    }

    const elapsedMs = now - progress.rateSampleTime;
    if (!force && elapsedMs < 400) return;

    const completedDelta = progress.completed - progress.rateSampleCompleted;
    if (elapsedMs > 0 && completedDelta > 0) {
      const currentRate = completedDelta * 1000 / elapsedMs;
      progress.stepsPerSecond = progress.stepsPerSecond == null
        ? currentRate
        : progress.stepsPerSecond * 0.65 + currentRate * 0.35;
    }

    progress.rateSampleTime = now;
    progress.rateSampleCompleted = progress.completed;
  }

  formatProgressStatus = function(progress) {
    const done = Math.min(progress.completed, progress.total);
    const percent = progress.total > 0 ? (done / progress.total) * 100 : 100;
    const parts = [
      `반복 ${done.toLocaleString('ko-KR')} / ${progress.total.toLocaleString('ko-KR')}`,
      `${percent.toFixed(1)}%`
    ];

    if (Number.isFinite(progress.stepsPerSecond)) {
      const rate = progress.stepsPerSecond;
      parts.push(`속도 ${rate >= 100 ? Math.round(rate).toLocaleString('ko-KR') : rate.toFixed(1)} step/s`);
    }

    if (progress.stack.length) {
      const path = progress.stack
        .map(frame => `${frame.name} ${frame.index + 1}/${frame.count}`)
        .join(' · ');
      parts.push(path);
    }
    return parts.join(' · ');
  };

  maybeYieldProgress = async function(progress, force = false) {
    if (!progress.total) return;
    const now = performance.now();
    updateStepRate(progress, now, force);
    if (!force && now - progress.lastYield < 80) return;
    workspaceStatus.textContent = formatProgressStatus(progress);
    await new Promise(resolve => setTimeout(resolve, 0));
    progress.lastYield = performance.now();
  };
})();

function resetDrawCanvas() { drawCtx.fillStyle = '#fff'; drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height); drawCtx.lineCap = 'round'; drawCtx.lineJoin = 'round'; drawCtx.lineWidth = 18; drawCtx.strokeStyle = '#111'; resetPreview(); }
function resetPreview() { previewCtx.fillStyle = '#000'; previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height); }
function pointerOnDraw(event) { const rect = drawCanvas.getBoundingClientRect(); return { x: (event.clientX - rect.left) * drawCanvas.width / rect.width, y: (event.clientY - rect.top) * drawCanvas.height / rect.height }; }
function findInkBounds(ctx, canvas) { const image = ctx.getImageData(0, 0, canvas.width, canvas.height); let minX = canvas.width, minY = canvas.height, maxX = -1, maxY = -1; for (let y = 0; y < canvas.height; y++) for (let x = 0; x < canvas.width; x++) { const i = (y * canvas.width + x) * 4, gray = (image.data[i] + image.data[i + 1] + image.data[i + 2]) / 3; if (gray < 245) { minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); } } return maxX < minX ? null : { minX, minY, maxX, maxY }; }
function preprocessCanvasTo28(canvas) { const ctx = canvas.getContext('2d', { willReadFrequently: true }), bounds = findInkBounds(ctx, canvas), temp = document.createElement('canvas'); temp.width = 28; temp.height = 28; const tctx = temp.getContext('2d', { willReadFrequently: true }); tctx.fillStyle = '#fff'; tctx.fillRect(0, 0, 28, 28); if (bounds) { const cropW = bounds.maxX - bounds.minX + 1, cropH = bounds.maxY - bounds.minY + 1, inner = 20, scale = Math.min(inner / cropW, inner / cropH), w = Math.max(1, Math.round(cropW * scale)), h = Math.max(1, Math.round(cropH * scale)), dx = Math.floor((28 - w) / 2), dy = Math.floor((28 - h) / 2); tctx.imageSmoothingEnabled = true; tctx.drawImage(canvas, bounds.minX, bounds.minY, cropW, cropH, dx, dy, w, h); } const small = tctx.getImageData(0, 0, 28, 28), data = new Float32Array(784); for (let i = 0; i < data.length; i++) { const p = i * 4, gray = (small.data[p] + small.data[p + 1] + small.data[p + 2]) / 3; data[i] = 1 - gray / 255; } return arrayValue(data, [28, 28]); }
function renderPreview(data) { const tiny = document.createElement('canvas'); tiny.width = 28; tiny.height = 28; const tctx = tiny.getContext('2d'), image = tctx.createImageData(28, 28); for (let i = 0; i < 784; i++) { const v = Math.round(clamp(data[i], 0, 1) * 255), p = i * 4; image.data[p] = v; image.data[p + 1] = v; image.data[p + 2] = v; image.data[p + 3] = 255; } tctx.putImageData(image, 0, 0); previewCtx.imageSmoothingEnabled = false; previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height); previewCtx.drawImage(tiny, 0, 0, previewCanvas.width, previewCanvas.height); }

for (const button of document.querySelectorAll('[data-block]')) button.addEventListener('click', () => addBlock(button.dataset.block));
evaluateBtn.addEventListener('click', evaluateGraph); resetWorkspaceBtn.addEventListener('click', resetWorkspace); deleteNodeBtn.addEventListener('click', deleteSelectedNode); clearDrawBtn.addEventListener('click', () => { resetDrawCanvas(); invalidatePreviews(); });
groupSelectBtn.addEventListener('click', startGroupSelection); createGroupBtn.addEventListener('click', createUserBlockFromSelection); cancelGroupBtn.addEventListener('click', cancelGroupSelection);
workspace.addEventListener('click', event => { if (event.target === workspace || event.target === nodesLayer || event.target === emptyState) cancelConnection(); }); window.addEventListener('resize', updateWires);
drawCanvas.addEventListener('pointerdown', event => { drawing = true; lastPoint = pointerOnDraw(event); drawCanvas.setPointerCapture?.(event.pointerId); });
drawCanvas.addEventListener('pointermove', event => { if (!drawing) return; const p = pointerOnDraw(event); drawCtx.beginPath(); drawCtx.moveTo(lastPoint.x, lastPoint.y); drawCtx.lineTo(p.x, p.y); drawCtx.stroke(); lastPoint = p; });
function stopDraw(event) { drawing = false; if (event?.pointerId != null && drawCanvas.hasPointerCapture?.(event.pointerId)) drawCanvas.releasePointerCapture(event.pointerId); invalidatePreviews(); }
drawCanvas.addEventListener('pointerup', stopDraw); drawCanvas.addEventListener('pointercancel', stopDraw);

loadUserBlocks(); renderMyBlocksPalette(); resetDrawCanvas(); resetWorkspace();
