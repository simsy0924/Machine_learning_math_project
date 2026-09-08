// Read-only dataset review. Compile only the selected prediction branch, cut its
// sampleImage inputs, and snapshot variables; never execute the training graph.
let datasetReviewRun = null;
let datasetReviewBusy = false;
let datasetReviewStop = false;
let datasetReviewPosition = 0;

function inspectReviewNode(node, nested = false, visiting = new Set()) {
  const def = getBlockDef(node.type);
  if (def.special || node.params?.manualBackpropMode === 'backward') {
    throw new Error('반복·값 변경·역전파가 없는 최종 확률/점수 출력 블록을 선택하세요.');
  }
  if (['sampleLabel', 'sampleOneHot', 'drawing', 'resize28', 'dataset', 'chooseClass', 'chooseSample', 'datasetSampleByIndex'].includes(node.type)
      || (nested && ['sampleImage', 'variable'].includes(node.type))) {
    throw new Error('그림값과 가중치 변수를 바깥에서 입력받는 예측 출력을 선택하세요. 정답 입력은 예측에 사용할 수 없습니다.');
  }
  if (node.type.startsWith('custom:')) {
    const id = node.type.slice(7);
    if (visiting.has(id)) throw new Error('사용자 블록이 순환 참조합니다.');
    visiting.add(id);
    for (const child of USER_BLOCKS.get(id).nodes) inspectReviewNode(child, true, visiting);
    visiting.delete(id);
  }
  return def;
}

function createDatasetReviewPredictor(outputId) {
  if (outputId == null) throw new Error('먼저 모델의 최종 확률/점수 출력 블록을 선택하세요.');
  const cache = ensureGraphStructureCache();
  const order = [], slots = new Map(), visiting = new Set();
  let imageCount = 0;
  const visit = id => {
    if (slots.has(id)) return slots.get(id);
    if (visiting.has(id)) throw new Error('순환 연결은 평가할 수 없습니다.');
    const source = graph.nodes.get(id);
    if (!source) throw new Error('존재하지 않는 출력 블록입니다.');
    visiting.add(id);
    const node = { id, type: source.type, params: JSON.parse(JSON.stringify(source.params || {})) };
    const def = inspectReviewNode(node);
    const step = { node, def, inputs: [] };
    if (node.type === 'sampleImage') {
      // Do not evaluate its dataset/index ancestors or change test_i.
      step.image = true;
      imageCount++;
    } else if (node.type === 'variable') {
      const stored = RUNTIME_VARIABLES.get(String(node.params.name || 'x'));
      step.constant = copyValue(stored?.signature === variableSignature(source) ? stored.value : initialVariableValue(node));
    } else {
      for (let i = 0; i < def.inputs.length; i++) {
        const connection = cachedGraphInput(cache, id, i);
        if (!connection) throw new Error(`입력 '${def.inputs[i]}'이 연결되지 않았습니다.`);
        step.inputs.push(visit(connection.from));
      }
    }
    visiting.delete(id);
    const slot = order.length;
    slots.set(id, slot);
    order.push(step);
    return slot;
  };
  const output = visit(outputId);
  if (!imageCount) throw new Error("'그림값'을 입력으로 사용하는 예측 출력을 선택하세요.");
  const libraryVersion = USER_BLOCK_LIBRARY_VERSION;
  return image => {
    if (libraryVersion !== USER_BLOCK_LIBRARY_VERSION) throw new Error('내 블록 정의가 변경되었습니다. 평가를 다시 시작하세요.');
    beginUserBlockExecutionGeneration();
    const values = [];
    for (const step of order) {
      values.push(step.image ? image : Object.hasOwn(step, 'constant') ? step.constant
        : step.def.compute(step.node, step.inputs.map(i => values[i])));
    }
    return values[output];
  };
}

function datasetReviewOutcome(value, classCount, truth) {
  if (!isArrayValue(value)) throw new Error('선택한 출력이 확률/점수 벡터가 아닙니다.');
  const vector = asArrayValue(value);
  if (vector.shape.length !== 1 || vector.data.length !== classCount) {
    throw new Error(`출력은 현재 데이터 ${classCount}종과 길이가 같은 벡터여야 합니다.`);
  }
  const scores = Array.from(vector.data);
  if (!scores.every(Number.isFinite)) throw new Error('출력에 NaN 또는 무한대가 있습니다. 가중치와 출력 블록을 확인하세요.');
  const best = Math.max(...scores);
  const winners = scores.flatMap((score, index) => score === best ? [index] : []);
  const credit = winners.includes(truth) ? 1 / winners.length : 0;
  return { scores, winners, credit, probability: isProbabilityVector(vector),
    status: winners.length > 1 ? 'tie' : credit === 1 ? 'correct' : 'wrong' };
}

function datasetReviewSample(entries, globalIndex) {
  const classIndex = globalIndex % entries.length;
  const index = Math.floor(globalIndex / entries.length);
  const entry = entries[classIndex];
  // Match datasetSampleByIndex ordering without silently wrapping at 10,000.
  if (index >= entry.imageCount) throw new Error(`평가 범위가 ${koreanClassName(entry.name)} 데이터 수를 넘습니다.`);
  return { classIndex, index, pixels: entry.bytes.subarray(index * 784, (index + 1) * 784) };
}

const reviewElement = id => document.getElementById(id);
function reviewText(tag, text, className) {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  return element;
}
function reviewScore(score, probability) {
  return probability ? `${(score * 100).toFixed(3)}%` : score.toFixed(6);
}
function filteredDatasetReviews(wrongOnly = false) {
  if (!datasetReviewRun) return [];
  const klass = reviewElement('reviewClass').value;
  const status = reviewElement('reviewFilter').value;
  return datasetReviewRun.rows.filter(row => (klass === '' || row.classIndex === Number(klass))
    && (wrongOnly ? row.credit === 0 : status === 'all' || (status === 'wrong' ? row.credit === 0 : row.status === status)));
}

function renderDatasetReviewImage() {
  const rows = filteredDatasetReviews();
  const target = reviewElement('reviewImage');
  target.replaceChildren();
  for (const id of ['reviewPrevious', 'reviewNext', 'reviewRandom']) reviewElement(id).disabled = !rows.length;
  reviewElement('reviewWrongRandom').disabled = !filteredDatasetReviews(true).length;
  if (!rows.length) {
    target.textContent = datasetReviewRun?.rows.length ? '현재 조건에 해당하는 그림이 없습니다.' : '평가 후 그림별 결과를 볼 수 있습니다.';
    return;
  }
  datasetReviewPosition = ((datasetReviewPosition % rows.length) + rows.length) % rows.length;
  const row = rows[datasetReviewPosition], run = datasetReviewRun;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 196;
  canvas.setAttribute('aria-label', `정답 ${koreanClassName(run.classes[row.classIndex])}, 데이터 번호 ${row.globalIndex}`);
  drawQuickDrawSample(canvas, datasetReviewSample(run.entries, row.globalIndex).pixels);
  target.append(reviewText('p', `${datasetReviewPosition + 1} / ${rows.length.toLocaleString()} · 데이터 #${row.globalIndex} · 종류 내 #${row.index}`, 'muted'), canvas);
  target.append(reviewText('p', `정답: ${koreanClassName(run.classes[row.classIndex])}`));
  target.append(reviewText('p', `예측: ${row.winners.map(i => koreanClassName(run.classes[i])).join(' / ')} · ${reviewScore(row.scores[row.winners[0]], row.probability)}`));
  const verdict = row.status === 'tie' ? `최고값 동률 ${row.winners.length}개 · 정답 점수 ${row.credit.toFixed(3)}` : row.credit ? '정답' : '오답';
  target.append(reviewText('strong', verdict, row.credit === 0 ? 'classification-error' : ''));
  const details = document.createElement('details');
  details.append(reviewText('summary', '모든 종류의 확률 / 점수'));
  const table = document.createElement('table');
  const body = document.createElement('tbody');
  row.scores.map((score, i) => ({ score, i })).sort((a, b) => b.score - a.score).forEach(({ score, i }) => {
    const tr = document.createElement('tr');
    tr.append(reviewText('td', `${koreanClassName(run.classes[i])}${i === row.classIndex ? ' (정답)' : ''}`), reviewText('td', reviewScore(score, row.probability)));
    body.append(tr);
  });
  table.append(body);details.append(table);target.append(details);
}

function renderDatasetReviewSummary(state) {
  const run = datasetReviewRun;
  if (!run) return;
  const accuracy = run.rows.length ? `${(run.credit / run.rows.length * 100).toFixed(2)}%` : '—';
  reviewElement('reviewSummary').textContent = `${state} · ${run.rows.length.toLocaleString()} / ${run.count.toLocaleString()}장 · 정확도 ${accuracy}`;
  reviewElement('reviewDetails').textContent = `정답 ${run.correct} · 오답 ${run.wrong} · 동률 ${run.ties} · 출력 #${run.outputId} · 시작 번호 ${run.start}. 실행 시작 시 가중치 기준이며, 동률은 정답 포함 시 1/동률 수로 계산합니다.`;
  const table = reviewElement('reviewClassSummary');table.replaceChildren();
  const header = document.createElement('tr');
  for (const title of ['정답 종류', '평가 수', '정확도']) header.append(reviewText('th', title));
  const head = document.createElement('thead');head.append(header);table.append(head);
  const body = document.createElement('tbody');
  run.stats.forEach((stat, i) => {
    const tr = document.createElement('tr');
    tr.append(reviewText('td', koreanClassName(run.classes[i])), reviewText('td', String(stat.count)), reviewText('td', stat.count ? `${(stat.credit / stat.count * 100).toFixed(2)}%` : '—'));
    body.append(tr);
  });
  table.append(body);
}

async function startDatasetReview() {
  if (datasetReviewBusy) return;
  const locked = [];
  try {
    if (evaluateBtn.disabled || evaluateSelectedBtn?.disabled || loadDatasetBtn.disabled) throw new Error('진행 중인 계산이나 데이터 불러오기가 끝난 뒤 평가하세요.');
    const classes = window.quickDrawDataset.getLoadedClassNames();
    if (!classes.length) throw new Error('Quick Draw 데이터를 먼저 불러오세요.');
    const entries = classes.map(name => datasetCache.get(name));
    const start = Number(reviewElement('reviewStart').value), count = Number(reviewElement('reviewCount').value);
    if (!Number.isSafeInteger(start) || start < 0 || !Number.isSafeInteger(count) || count < 1) throw new Error('시작 번호는 0 이상, 평가 수는 1 이상의 정수여야 합니다.');
    const total = Math.min(...entries.map(e => e.imageCount), 10000) * classes.length;
    if (start + count > total) throw new Error(`중복 없이 평가할 수 있는 번호는 0~${total - 1}입니다. 시작 번호와 평가 수를 줄여 주세요.`);
    const predict = createDatasetReviewPredictor(selectedNodeId);
    // Validate shape/finite values before replacing the user's last review.
    const first = datasetReviewSample(entries, start);
    const input = pixels => arrayValue(Float32Array.from(pixels, v => v / 255), [28, 28]);
    const firstOutcome = datasetReviewOutcome(predict(input(first.pixels)), classes.length, first.classIndex);
    datasetReviewRun = { classes, entries, start, count, outputId: selectedNodeId, rows: [], credit: 0, correct: 0, wrong: 0, ties: 0,
      stats: classes.map(() => ({ count: 0, credit: 0 })) };
    datasetReviewBusy = true;datasetReviewStop = false;datasetReviewPosition = 0;
    for (const id of ['reviewRun', 'reviewStart', 'reviewCount', 'evaluateBtn', 'evaluateSelectedBtn', 'loadDatasetBtn', 'classifyDrawingBtn']) {
      const el = reviewElement(id);if (el) { locked.push([el, el.disabled]);el.disabled = true; }
    }
    reviewElement('reviewStop').disabled = false;
    const classFilter = reviewElement('reviewClass');classFilter.replaceChildren(new Option('모든 종류', ''));
    classes.forEach((name, i) => classFilter.add(new Option(koreanClassName(name), String(i))));
    reviewElement('reviewFilter').value = 'all';
    let lastYield = performance.now();
    for (let offset = 0; offset < count; offset++) {
      if (datasetReviewStop) break;
      const globalIndex = start + offset;
      const sample = datasetReviewSample(entries, globalIndex);
      const outcome = offset === 0 ? firstOutcome : datasetReviewOutcome(predict(input(sample.pixels)), classes.length, sample.classIndex);
      const run = datasetReviewRun;
      run.rows.push({ globalIndex, classIndex: sample.classIndex, index: sample.index, ...outcome });
      run.credit += outcome.credit;
      if (outcome.status === 'tie') run.ties++;
      else if (outcome.credit) run.correct++;
      else run.wrong++;
      run.stats[sample.classIndex].count++;run.stats[sample.classIndex].credit += outcome.credit;
      if (performance.now() - lastYield > 50) {
        renderDatasetReviewSummary('평가 중');renderDatasetReviewImage();
        await new Promise(resolve => setTimeout(resolve, 0));lastYield = performance.now();
      }
    }
    renderDatasetReviewSummary(datasetReviewStop ? '중단 · 처리한 그림 기준' : '평가 완료');
    renderDatasetReviewImage();
  } catch (error) {
    if (datasetReviewBusy) { renderDatasetReviewSummary('오류로 중단 · 처리한 그림 기준');renderDatasetReviewImage(); }
    const target = reviewElement('reviewSummary');
    target.textContent += `\n오류: ${error.message}`;
  } finally {
    datasetReviewBusy = false;
    for (const [el, disabled] of locked) el.disabled = disabled;
    reviewElement('reviewStop').disabled = true;
  }
}

function moveDatasetReview(delta) { datasetReviewPosition += delta;renderDatasetReviewImage(); }
function randomDatasetReview(wrongOnly = false) {
  const previous = filteredDatasetReviews()[datasetReviewPosition]?.globalIndex;
  if (wrongOnly) reviewElement('reviewFilter').value = 'wrong';
  const rows = filteredDatasetReviews();
  // Uniform among eligible images; exclude the current image if alternatives exist.
  const previousIndex = rows.findIndex(row => row.globalIndex === previous);
  if (rows.length > 1 && previousIndex >= 0) {
    datasetReviewPosition = (previousIndex + 1 + Math.floor(Math.random() * (rows.length - 1))) % rows.length;
  } else datasetReviewPosition = Math.floor(Math.random() * rows.length) || 0;
  renderDatasetReviewImage();
}
reviewElement('reviewRun').addEventListener('click', startDatasetReview);
reviewElement('reviewStop').addEventListener('click', () => { datasetReviewStop = true; });
reviewElement('reviewPrevious').addEventListener('click', () => moveDatasetReview(-1));
reviewElement('reviewNext').addEventListener('click', () => moveDatasetReview(1));
reviewElement('reviewRandom').addEventListener('click', () => randomDatasetReview());
reviewElement('reviewWrongRandom').addEventListener('click', () => randomDatasetReview(true));
for (const id of ['reviewClass', 'reviewFilter']) reviewElement(id).addEventListener('change', () => { datasetReviewPosition = 0;renderDatasetReviewImage(); });
