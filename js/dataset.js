// Quick Draw dataset loading and the sample gallery.
// Only the classes the user ticks are fetched; each .bin file is a flat run of
// 28×28 grayscale bytes.

const datasetCache = new Map();
const classPicker = document.getElementById('classPicker');
const loadDatasetBtn = document.getElementById('loadDatasetBtn');
const selectAllClassesBtn = document.getElementById('selectAllClassesBtn');
const clearClassSelectionBtn = document.getElementById('clearClassSelectionBtn');
const randomizeSamplesBtn = document.getElementById('randomizeSamplesBtn');
const datasetStatus = document.getElementById('datasetStatus');
const datasetCount = document.getElementById('datasetCount');
const datasetSamples = document.getElementById('datasetSamples');

function datasetUrl(name) {
  if (location.protocol === 'file:') return `https://raw.githubusercontent.com/simsy0924/Machine_learning_math_project/main/data/${name}.bin`;
  return `data/${name}.bin`;
}

function selectedClassNames() {
  return Array.from(classPicker.querySelectorAll('input[type="checkbox"]:checked')).map(input => input.value);
}

function renderClassPicker() {
  const defaults = new Set(['cat', 'fish', 'house']);
  classPicker.replaceChildren();
  for (const [name, label] of QUICK_DRAW_CLASSES) {
    const item = document.createElement('label');
    item.className = 'class-option';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.value = name;
    checkbox.checked = defaults.has(name);
    const text = document.createElement('span');
    text.textContent = label;
    item.append(checkbox, text);
    classPicker.appendChild(item);
  }
}

async function loadClass(name) {
  if (datasetCache.has(name)) return datasetCache.get(name);
  const response = await fetch(datasetUrl(name), { cache: 'force-cache' });
  if (!response.ok) throw new Error(`${name}.bin 불러오기 실패 (${response.status})`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.length % 784 !== 0) throw new Error(`${name}.bin 크기가 784의 배수가 아닙니다.`);
  const entry = { name, bytes, imageCount: bytes.length / 784 };
  datasetCache.set(name, entry);
  return entry;
}

function getImage(name, index) {
  const entry = datasetCache.get(name);
  if (!entry) throw new Error(`${name} 데이터가 아직 로드되지 않았습니다.`);
  if (!Number.isInteger(index) || index < 0 || index >= entry.imageCount) {
    throw new RangeError(`이미지 번호는 0~${entry.imageCount - 1} 사이여야 합니다.`);
  }
  const start = index * 784;
  return entry.bytes.subarray(start, start + 784);
}

function getNormalizedImage(name, index) {
  const pixels = getImage(name, index), result = new Float32Array(784);
  for (let i = 0; i < 784; i++) result[i] = pixels[i] / 255;
  return result;
}

function drawQuickDrawSample(canvas, pixels) {
  const tiny = document.createElement('canvas');
  tiny.width = 28;
  tiny.height = 28;
  const tinyCtx = tiny.getContext('2d'), image = tinyCtx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const value = pixels[i], p = i * 4;
    image.data[p] = value; image.data[p + 1] = value; image.data[p + 2] = value; image.data[p + 3] = 255;
  }
  tinyCtx.putImageData(image, 0, 0);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(tiny, 0, 0, canvas.width, canvas.height);
}

function updateDatasetSummary() {
  const loaded = Array.from(datasetCache.values());
  const totalImages = loaded.reduce((sum, entry) => sum + entry.imageCount, 0);
  const totalBytes = loaded.reduce((sum, entry) => sum + entry.bytes.byteLength, 0);
  datasetCount.textContent = `${loaded.length}개 로드`;
  if (!loaded.length) {
    datasetStatus.textContent = '아직 불러온 데이터가 없습니다.';
    randomizeSamplesBtn.disabled = true;
    return;
  }
  datasetStatus.textContent = `${loaded.length}종 · ${totalImages.toLocaleString()}장 · ${(totalBytes / 1_000_000).toFixed(1)} MB`;
  randomizeSamplesBtn.disabled = false;
}

function renderRandomSamples() {
  datasetSamples.replaceChildren();
  const loaded = Array.from(datasetCache.values());
  if (!loaded.length) {
    const empty = document.createElement('p');
    empty.className = 'muted dataset-empty';
    empty.textContent = '데이터를 불러오면 여기에 실제 샘플이 표시됩니다.';
    datasetSamples.appendChild(empty);
    return;
  }
  for (const entry of loaded) {
    const label = koreanClassName(entry.name);
    const index = Math.floor(Math.random() * entry.imageCount);
    const card = document.createElement('div');
    card.className = 'sample-card';
    const canvas = document.createElement('canvas');
    canvas.width = 84;
    canvas.height = 84;
    drawQuickDrawSample(canvas, getImage(entry.name, index));
    const caption = document.createElement('div');
    const strong = document.createElement('strong');
    strong.textContent = label;
    const span = document.createElement('span');
    span.textContent = `#${index}`;
    caption.append(strong, span);
    card.append(canvas, caption);
    datasetSamples.appendChild(card);
  }
}

async function loadSelectedClasses() {
  const selected = selectedClassNames();
  if (!selected.length) { datasetStatus.textContent = '최소 한 종류를 선택하세요.'; return; }
  loadDatasetBtn.disabled = true;
  randomizeSamplesBtn.disabled = true;
  try {
    for (const name of Array.from(datasetCache.keys())) if (!selected.includes(name)) datasetCache.delete(name);
    for (let i = 0; i < selected.length; i++) {
      const name = selected[i];
      datasetStatus.textContent = `${koreanClassName(name)} 불러오는 중... (${i + 1}/${selected.length})`;
      await loadClass(name);
    }
    updateDatasetSummary();
    renderRandomSamples();
  } catch (error) {
    console.error(error);
    datasetStatus.textContent = `오류: ${error.message}`;
  } finally {
    loadDatasetBtn.disabled = false;
    randomizeSamplesBtn.disabled = datasetCache.size === 0;
  }
}

selectAllClassesBtn.addEventListener('click', () => classPicker.querySelectorAll('input[type="checkbox"]').forEach(input => { input.checked = true; }));
clearClassSelectionBtn.addEventListener('click', () => classPicker.querySelectorAll('input[type="checkbox"]').forEach(input => { input.checked = false; }));
loadDatasetBtn.addEventListener('click', loadSelectedClasses);
randomizeSamplesBtn.addEventListener('click', renderRandomSamples);
renderClassPicker();
updateDatasetSummary();
renderRandomSamples();

window.quickDrawDataset = {
  classes: QUICK_DRAW_CLASSES.map(([name, label]) => ({ name, label })),
  cache: datasetCache,
  loadSelectedClasses,
  selectedClassNames,
  getImage,
  getNormalizedImage,
  getLoadedClassNames: () => Array.from(datasetCache.keys()),
  getClassLabel: koreanClassName
};
