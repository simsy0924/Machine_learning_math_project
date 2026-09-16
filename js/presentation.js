// Read-only HTTP loading; exportProject downloads a separate .mmlab copy.
const presentationToggle = document.getElementById('presentationToggle');
const presentationPrevious = document.getElementById('presentationPrevious');
const presentationNext = document.getElementById('presentationNext');
let presentationSteps = [];
let presentationDescriptions = {};
let presentationIndex = -1;
let presentationEditorSnapshot = null;
const PRESENTATION_ROOT = new URL('presentation/', document.baseURI);
const presentationPanelIds = ['workspaceStatus', 'selectedEvaluationResult', 'drawingClassificationResult',
  'reviewSummary', 'reviewDetails', 'reviewClassSummary', 'reviewClass'];

function capturePresentationWorkspace() {
  return {
    project: buildProjectSnapshot(),
    dataset: new Map(datasetCache),
    selected: selectedNodeId,
    selection: { pendingOutput, selectedConnection: selectedConnection && { ...selectedConnection }, groupSelectionMode, ids: [...groupSelectedIds] },
    strokes: structuredClone(rawStrokes),
    drawing: drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height),
    preview: previewCtx.getImageData(0, 0, previewCanvas.width, previewCanvas.height),
    review: datasetReviewRun, reviewPosition: datasetReviewPosition,
    panels: presentationPanelIds.map(id => [id, document.getElementById(id).innerHTML]),
    reviewInputs: ['reviewStart', 'reviewCount', 'reviewFilter', 'reviewClass'].map(id => [id, document.getElementById(id).value]),
    results: Array.from(graph.nodes.values()).map(n => [n.id, n.lastValue === undefined ? undefined : copyValue(n.lastValue), n.lastError])
  };
}

async function restorePresentationWorkspace(saved) {
  await restoreProject(saved.project, { loadDataset: false });
  datasetCache.clear();
  for (const [name, entry] of saved.dataset) datasetCache.set(name, entry);
  updateDatasetSummary(); renderRandomSamples();
  rawStrokes.splice(0, rawStrokes.length, ...structuredClone(saved.strokes));
  activeStroke = null; drawingPointerDown = false;
  drawCtx.putImageData(saved.drawing, 0, 0); previewCtx.putImageData(saved.preview, 0, 0);
  selectedNodeId = saved.selected;
  for (const [id, value, error] of saved.results) {
    const node = graph.nodes.get(id);
    if (node) { node.lastValue = value; node.lastError = error; updateNodePreview(node); }
  }
  pendingOutput = saved.selection.pendingOutput;
  selectedConnection = saved.selection.selectedConnection;
  if (saved.selection.groupSelectionMode) {
    startGroupSelection();
    for (const id of saved.selection.ids) {
      groupSelectedIds.add(id);
      nodesLayer.querySelector(`[data-node-id="${id}"]`)?.classList.add('group-selected');
    }
    updateGroupUI();
  }
  nodesLayer.querySelectorAll('.node').forEach(el => el.classList.toggle('selected', Number(el.dataset.nodeId) === selectedNodeId));
  updateWires(); renderInspector();
  datasetReviewRun = saved.review; datasetReviewPosition = saved.reviewPosition;
  for (const [id, html] of saved.panels) document.getElementById(id).innerHTML = html;
  for (const [id, value] of saved.reviewInputs) document.getElementById(id).value = value;
  renderDatasetReviewImage();
}

function clearPresentationExtras() {
  datasetCache.clear(); updateDatasetSummary(); renderRandomSamples(); resetDrawCanvas();
  datasetReviewRun = null; datasetReviewPosition = 0; datasetReviewStop = false;
  document.getElementById('reviewClass').replaceChildren(new Option('모든 종류', ''));
  document.getElementById('reviewFilter').value = 'all';
  document.getElementById('reviewClassSummary').replaceChildren();
  for (const id of ['reviewSummary', 'reviewDetails', 'selectedEvaluationResult', 'drawingClassificationResult']) {
    document.getElementById(id).textContent = '이 단계에서 아직 실행하지 않았습니다.';
  }
  renderDatasetReviewImage();
}

function presentationAssetUrl(path) {
  if (typeof path !== 'string' || !path.trim()) throw new Error('발표 파일 경로가 없습니다.');
  const url = new URL(path, PRESENTATION_ROOT);
  if (url.origin !== PRESENTATION_ROOT.origin || !url.pathname.startsWith(PRESENTATION_ROOT.pathname)) {
    throw new Error('발표 파일은 presentation 폴더 안에 두세요.');
  }
  return url;
}

async function fetchPresentationJson(path, limit = MAX_PROJECT_FILE_BYTES) {
  const response = await fetch(presentationAssetUrl(path), { cache: 'no-store', signal: AbortSignal.timeout(30000) });
  if (!response.ok) throw new Error(`${path} 불러오기 실패 (${response.status})`);
  if (Number(response.headers.get('content-length')) > limit) throw new Error(`${path} 파일이 너무 큽니다.`);
  const blob = await response.blob();
  if (blob.size > limit) throw new Error(`${path} 파일이 너무 큽니다.`);
  try { return JSON.parse(await blob.text()); }
  catch { throw new Error(`${path}의 JSON 형식을 확인하세요.`); }
}

async function readPresentationManifest() {
  const manifest = await fetchPresentationJson('steps.json', 1024 * 1024);
  if (!Array.isArray(manifest.steps) || !manifest.steps.length) throw new Error('steps.json에 단계를 추가하세요.');
  const ids = new Set();
  for (const step of manifest.steps) {
    if (!step || typeof step.id !== 'string' || !step.id || ids.has(step.id) || typeof step.file !== 'string' || !step.file.endsWith('.mmlab')) {
      throw new Error('각 단계에 고유한 id와 .mmlab 파일 경로가 필요합니다.');
    }
    ids.add(step.id); presentationAssetUrl(step.file);
  }
  const descriptions = await fetchPresentationJson('descriptions.json', 1024 * 1024);
  if (!descriptions || Array.isArray(descriptions) || typeof descriptions !== 'object' || Object.values(descriptions).some(v => typeof v !== 'string')) {
    throw new Error('descriptions.json은 단계 id와 설명 문자열로 작성하세요.');
  }
  return { steps: manifest.steps, descriptions };
}

function updatePresentationControls() {
  presentationToggle.textContent = presentationActive ? '발표 모드 ON' : '발표 모드 OFF · 편집 모드';
  presentationToggle.setAttribute('aria-pressed', String(presentationActive));
  presentationToggle.disabled = presentationBusy;
  document.getElementById('presentationControls').hidden = !presentationActive;
  presentationPrevious.disabled = presentationBusy || presentationIndex <= 0;
  presentationNext.disabled = presentationBusy || presentationIndex >= presentationSteps.length - 1;
  const step = presentationSteps[presentationIndex];
  document.getElementById('presentationStep').textContent = step ? `${presentationIndex + 1} / ${presentationSteps.length} · ${step.title || step.id}` : '';
  const description = step ? presentationDescriptions[step.id] || '' : '';
  document.getElementById('presentationDescription').hidden = !presentationActive || !description;
  document.getElementById('presentationDescriptionText').textContent = description;
}

function assertPresentationIdle() {
  if (autosaveSuspendDepth > 0 || evaluateBtn.disabled || evaluateSelectedBtn.disabled || loadDatasetBtn.disabled || datasetReviewBusy || importProjectBtn.disabled || importWeightsBtn.disabled || drawingPointerDown) {
    throw new Error('진행 중인 계산이나 불러오기가 끝난 뒤 이동하세요.');
  }
  if (isUserBlockWorkspaceEditing() || window.isManualBackpropWorkspaceEditing?.()) {
    throw new Error('블록 내부 편집을 마친 뒤 이동하세요.');
  }
}

async function changePresentation(target) {
  if (presentationBusy) return;
  const status = document.getElementById('presentationStatus');
  let rollback = null;
  let entering = false;
  let mutated = false;
  try {
    assertPresentationIdle();
    const leaving = target === 'off';
    entering = !presentationActive && !leaving;
    if (leaving && !presentationActive) return;
    if (entering) saveWorkspaceNow();
    presentationBusy = true;
    // Freeze interaction throughout async restore, including keyboard deletion.
    document.querySelector('.app-shell').inert = true;
    document.querySelector('.top-actions').inert = true;
    updatePresentationControls();
    status.hidden = false; status.textContent = '불러오는 중…';
    if (leaving) {
      rollback = capturePresentationWorkspace();
      mutated = true;
      await restorePresentationWorkspace(presentationEditorSnapshot);
      presentationActive = false; presentationIndex = -1; presentationEditorSnapshot = null;
    } else {
      const config = entering ? await readPresentationManifest() : { steps: presentationSteps, descriptions: presentationDescriptions };
      const index = entering ? 0 : target;
      if (!Number.isInteger(index) || index < 0 || index >= config.steps.length) throw new Error('없는 단계입니다.');
      const project = await fetchPresentationJson(config.steps[index].file);
      validateProjectSnapshot(project);
      rollback = capturePresentationWorkspace();
      mutated = true;
      clearPresentationExtras();
      await restoreProject(project);
      if ((project.datasetSelection || []).some(name => !datasetCache.has(name))) {
        throw new Error('단계에 필요한 데이터를 모두 불러오지 못했습니다. 다시 시도하세요.');
      }
      if (entering) presentationEditorSnapshot = rollback;
      presentationSteps = config.steps; presentationDescriptions = config.descriptions;
      presentationIndex = index; presentationActive = true;
    }
    status.hidden = true;
  } catch (error) {
    if (mutated && rollback) {
      try { await restorePresentationWorkspace(rollback); }
      catch (restoreError) { console.error('화면 복구 실패', restoreError); }
    }
    status.hidden = false; status.textContent = `전환 실패: ${error.message}`;
  } finally {
    presentationBusy = false;
    document.querySelector('.app-shell').inert = false;
    document.querySelector('.top-actions').inert = false;
    updatePresentationControls();
  }
}

window.addEventListener('keydown', event => {
  if (presentationBusy) { event.preventDefault(); event.stopImmediatePropagation(); }
}, true);
presentationToggle.addEventListener('click', () => changePresentation(presentationActive ? 'off' : 0));
presentationPrevious.addEventListener('click', () => changePresentation(presentationIndex - 1));
presentationNext.addEventListener('click', () => changePresentation(presentationIndex + 1));
