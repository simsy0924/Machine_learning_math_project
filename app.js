const drawCanvas = document.getElementById('drawCanvas');
const drawCtx = drawCanvas.getContext('2d', { willReadFrequently: true });
const previewCanvas = document.getElementById('previewCanvas');
const previewCtx = previewCanvas.getContext('2d');

const clearBtn = document.getElementById('clearBtn');
const processBtn = document.getElementById('processBtn');
const copyBtn = document.getElementById('copyBtn');
const vectorOutput = document.getElementById('vectorOutput');
const statusEl = document.getElementById('status');
const minValEl = document.getElementById('minVal');
const maxValEl = document.getElementById('maxVal');
const meanValEl = document.getElementById('meanVal');

let drawing = false;
let lastX = 0;
let lastY = 0;
let currentVector = new Float32Array(784);

function resetDrawCanvas() {
  drawCtx.fillStyle = '#ffffff';
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  drawCtx.lineCap = 'round';
  drawCtx.lineJoin = 'round';
  drawCtx.strokeStyle = '#000000';
  drawCtx.lineWidth = 24;
}

function resetPreview() {
  previewCtx.fillStyle = '#000';
  previewCtx.fillRect(0, 0, 280, 280);
}

function getPointerPos(event) {
  const rect = drawCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (drawCanvas.width / rect.width),
    y: (event.clientY - rect.top) * (drawCanvas.height / rect.height)
  };
}

function startDrawing(event) {
  drawing = true;
  const p = getPointerPos(event);
  lastX = p.x;
  lastY = p.y;
  drawCanvas.setPointerCapture?.(event.pointerId);
}

function draw(event) {
  if (!drawing) return;
  const p = getPointerPos(event);
  drawCtx.beginPath();
  drawCtx.moveTo(lastX, lastY);
  drawCtx.lineTo(p.x, p.y);
  drawCtx.stroke();
  lastX = p.x;
  lastY = p.y;
}

function stopDrawing(event) {
  drawing = false;
  if (event?.pointerId != null && drawCanvas.hasPointerCapture?.(event.pointerId)) {
    drawCanvas.releasePointerCapture(event.pointerId);
  }
}

function findInkBounds(imageData) {
  const { data, width, height } = imageData;
  let minX = width, minY = height, maxX = -1, maxY = -1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
      if (gray < 245) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < minX || maxY < minY) return null;
  return { minX, minY, maxX, maxY };
}

// Quick Draw의 28x28 bitmap과 입력 분포를 최대한 비슷하게 만들기 위한 전처리.
// 1) 잉크 bounding box를 찾음
// 2) 가로/세로 비율을 유지한 채 20x20 안에 맞춤
// 3) 28x28 중앙에 배치
// 4) 검은 선을 1, 흰 배경을 0으로 변환
function preprocessTo28() {
  const src = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
  const bounds = findInkBounds(src);
  if (!bounds) return new Float32Array(784);

  const cropW = bounds.maxX - bounds.minX + 1;
  const cropH = bounds.maxY - bounds.minY + 1;
  const inner = 20;
  const scale = Math.min(inner / cropW, inner / cropH);
  const destW = Math.max(1, Math.round(cropW * scale));
  const destH = Math.max(1, Math.round(cropH * scale));
  const dx = Math.floor((28 - destW) / 2);
  const dy = Math.floor((28 - destH) / 2);

  const temp = document.createElement('canvas');
  temp.width = 28;
  temp.height = 28;
  const tctx = temp.getContext('2d', { willReadFrequently: true });
  tctx.fillStyle = '#fff';
  tctx.fillRect(0, 0, 28, 28);
  tctx.imageSmoothingEnabled = true;
  tctx.drawImage(
    drawCanvas,
    bounds.minX, bounds.minY, cropW, cropH,
    dx, dy, destW, destH
  );

  const small = tctx.getImageData(0, 0, 28, 28);
  const vector = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    const p = i * 4;
    const gray = (small.data[p] + small.data[p + 1] + small.data[p + 2]) / 3;
    vector[i] = 1 - gray / 255;
  }
  return vector;
}

function renderPreview(vector) {
  const image = previewCtx.createImageData(28, 28);
  for (let i = 0; i < vector.length; i++) {
    const v = Math.round(Math.max(0, Math.min(1, vector[i])) * 255);
    const p = i * 4;
    image.data[p] = v;
    image.data[p + 1] = v;
    image.data[p + 2] = v;
    image.data[p + 3] = 255;
  }

  const tiny = document.createElement('canvas');
  tiny.width = 28;
  tiny.height = 28;
  tiny.getContext('2d').putImageData(image, 0, 0);

  previewCtx.imageSmoothingEnabled = false;
  previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  previewCtx.drawImage(tiny, 0, 0, previewCanvas.width, previewCanvas.height);
}

function updateVectorText(vector) {
  const rows = [];
  for (let y = 0; y < 28; y++) {
    const row = [];
    for (let x = 0; x < 28; x++) {
      row.push(vector[y * 28 + x].toFixed(2));
    }
    rows.push(row.join(', '));
  }
  vectorOutput.value = '[\n' + rows.map(r => '  ' + r).join(',\n') + '\n]';

  let min = Infinity, max = -Infinity, sum = 0;
  for (const v of vector) {
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  minValEl.textContent = min.toFixed(3);
  maxValEl.textContent = max.toFixed(3);
  meanValEl.textContent = (sum / vector.length).toFixed(3);
}

function processDrawing() {
  currentVector = preprocessTo28();
  renderPreview(currentVector);
  updateVectorText(currentVector);
  statusEl.textContent = '784차원 변환 완료';
}

clearBtn.addEventListener('click', () => {
  resetDrawCanvas();
  resetPreview();
  currentVector = new Float32Array(784);
  updateVectorText(currentVector);
  statusEl.textContent = '준비';
});

processBtn.addEventListener('click', processDrawing);
copyBtn.addEventListener('click', async () => {
  await navigator.clipboard.writeText(JSON.stringify(Array.from(currentVector)));
  const old = copyBtn.textContent;
  copyBtn.textContent = '복사됨';
  setTimeout(() => copyBtn.textContent = old, 900);
});

drawCanvas.addEventListener('pointerdown', startDrawing);
drawCanvas.addEventListener('pointermove', draw);
drawCanvas.addEventListener('pointerup', stopDrawing);
drawCanvas.addEventListener('pointercancel', stopDrawing);
drawCanvas.addEventListener('pointerleave', (e) => {
  if (e.buttons === 0) stopDrawing(e);
});

resetDrawCanvas();
resetPreview();
updateVectorText(currentVector);
