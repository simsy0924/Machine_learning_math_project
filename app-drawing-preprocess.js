// Match hand-drawn inference more closely to the Quick, Draw! bitmap pipeline.
//
// Quick Draw's simplified drawings are uniformly scaled and the 28x28 bitmaps
// are centered on the drawing bounding box. The previous hand-drawing path used
// an MNIST-like 20x20 inner box, leaving much more margin than the training
// bitmaps. That distribution shift can make otherwise valid drawings look unlike
// the samples used to train the model.

(function installQuickDrawDrawingPreprocess() {
  const OUTPUT_SIZE = 28;
  const TARGET_SPAN = 26; // one-pixel safety margin for antialiased stroke edges
  const DRAW_LINE_WIDTH = 10;
  const INK_GAMMA = 0.85; // gently lift antialiased gray pixels without changing geometry

  preprocessCanvasTo28 = function(canvas) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const bounds = findInkBounds(ctx, canvas);
    const temp = document.createElement('canvas');
    temp.width = OUTPUT_SIZE;
    temp.height = OUTPUT_SIZE;

    const tctx = temp.getContext('2d', { willReadFrequently: true });
    tctx.fillStyle = '#fff';
    tctx.fillRect(0, 0, OUTPUT_SIZE, OUTPUT_SIZE);

    if (bounds) {
      const cropW = bounds.maxX - bounds.minX + 1;
      const cropH = bounds.maxY - bounds.minY + 1;
      const scale = Math.min(TARGET_SPAN / cropW, TARGET_SPAN / cropH);
      const width = Math.max(1, cropW * scale);
      const height = Math.max(1, cropH * scale);
      const dx = (OUTPUT_SIZE - width) / 2;
      const dy = (OUTPUT_SIZE - height) / 2;

      tctx.imageSmoothingEnabled = true;
      if ('imageSmoothingQuality' in tctx) tctx.imageSmoothingQuality = 'high';
      tctx.drawImage(
        canvas,
        bounds.minX,
        bounds.minY,
        cropW,
        cropH,
        dx,
        dy,
        width,
        height
      );
    }

    const small = tctx.getImageData(0, 0, OUTPUT_SIZE, OUTPUT_SIZE);
    const data = new Float32Array(OUTPUT_SIZE * OUTPUT_SIZE);
    for (let i = 0; i < data.length; i++) {
      const p = i * 4;
      const gray = (small.data[p] + small.data[p + 1] + small.data[p + 2]) / 3;
      const ink = 1 - gray / 255;
      data[i] = ink <= 0 ? 0 : Math.pow(ink, INK_GAMMA);
    }
    return arrayValue(data, [OUTPUT_SIZE, OUTPUT_SIZE]);
  };

  // The old 18px brush became relatively thick after bounding-box scaling.
  // A thinner brush better matches the stroke density of the 28x28 Quick Draw
  // bitmaps while keeping the same drawing interaction.
  resetDrawCanvas = function() {
    drawCtx.fillStyle = '#fff';
    drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    drawCtx.lineCap = 'round';
    drawCtx.lineJoin = 'round';
    drawCtx.lineWidth = DRAW_LINE_WIDTH;
    drawCtx.strokeStyle = '#111';
    resetPreview();
  };

  // app-boot.js has already initialized the canvas before this override loads.
  // Reinitialize once so the new brush width is active immediately.
  resetDrawCanvas();
})();
