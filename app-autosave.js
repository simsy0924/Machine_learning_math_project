// Lightweight browser autosave for editing progress.
// Deliberately excludes RUNTIME_VARIABLES: trained weights, gradient accumulators
// and other live training state are never written by this module.

(function installWorkspaceAutosave() {
  const AUTOSAVE_KEY = 'machine-learning-math-project.workspace-autosave.v1';
  const AUTOSAVE_DELAY_MS = 220;
  let timer = null;
  let restoring = false;

  function buildAutosaveSnapshot() {
    const datasetSelection = window.quickDrawDataset?.selectedClassNames?.() ||
      Array.from(document.querySelectorAll('#classPicker input[type="checkbox"]:checked')).map(input => input.value);

    return {
      format: PROJECT_FORMAT,
      version: PROJECT_VERSION,
      autosave: true,
      savedAt: new Date().toISOString(),
      graph: {
        nodes: Array.from(graph.nodes.values()).map(serializeNode),
        connections: graph.connections.map(connection => ({ ...connection }))
      },
      userBlocks: Array.from(USER_BLOCKS.values()).map(definition => JSON.parse(JSON.stringify(definition))),
      // Keep restoreProject compatible while explicitly storing no live variables.
      runtimeVariables: [],
      datasetSelection,
      viewport: typeof getWorkspaceViewSnapshot === 'function' ? getWorkspaceViewSnapshot() : undefined
    };
  }

  function saveNow() {
    if (restoring) return;
    if (timer != null) {
      clearTimeout(timer);
      timer = null;
    }
    try {
      localStorage.setItem(AUTOSAVE_KEY, JSON.stringify(buildAutosaveSnapshot()));
    } catch (error) {
      console.warn('작업공간 자동 저장 실패', error);
    }
  }

  function scheduleSave() {
    if (restoring) return;
    if (timer != null) clearTimeout(timer);
    timer = setTimeout(saveNow, AUTOSAVE_DELAY_MS);
  }

  // Wrap persistent graph mutations. The underlying functions remain the single
  // source of truth; autosave merely observes that a mutation finished.
  const addBlockBeforeAutosave = addBlock;
  addBlock = function(...args) {
    const result = addBlockBeforeAutosave(...args);
    scheduleSave();
    return result;
  };

  const finishConnectionBeforeAutosave = finishConnection;
  finishConnection = function(...args) {
    const result = finishConnectionBeforeAutosave(...args);
    scheduleSave();
    return result;
  };

  const removeNodesBeforeAutosave = removeNodes;
  removeNodes = function(...args) {
    const result = removeNodesBeforeAutosave(...args);
    scheduleSave();
    return result;
  };

  const createUserBlockBeforeAutosave = createUserBlockFromSelection;
  createUserBlockFromSelection = function(...args) {
    const result = createUserBlockBeforeAutosave(...args);
    scheduleSave();
    return result;
  };

  if (typeof deleteUserBlock === 'function') {
    const deleteUserBlockBeforeAutosave = deleteUserBlock;
    deleteUserBlock = function(...args) {
      const deleted = deleteUserBlockBeforeAutosave(...args);
      if (deleted) scheduleSave();
      return deleted;
    };
  }

  const resetWorkspaceBeforeAutosave = resetWorkspace;
  resetWorkspace = function(...args) {
    const result = resetWorkspaceBeforeAutosave(...args);
    scheduleSave();
    return result;
  };

  // A manual project import should become the new autosave baseline, while the
  // restore itself must not briefly save the empty workspace used during rebuild.
  const restoreProjectBeforeAutosave = restoreProject;
  restoreProject = async function(project) {
    const wasRestoring = restoring;
    restoring = true;
    try {
      await restoreProjectBeforeAutosave(project);
    } finally {
      restoring = wasRestoring;
    }
    if (!restoring) scheduleSave();
  };

  // Inspector controls mutate node.params inside their own input listener.
  inspectorControls.addEventListener('input', scheduleSave);
  inspectorControls.addEventListener('change', scheduleSave);

  // Dataset selection, node dragging, panning and zooming are state changes too.
  document.getElementById('classPicker')?.addEventListener('change', scheduleSave);
  document.getElementById('selectAllClassesBtn')?.addEventListener('click', scheduleSave);
  document.getElementById('clearClassSelectionBtn')?.addEventListener('click', scheduleSave);
  // app-boot registered the reset button before this module replaced the global
  // resetWorkspace binding, so observe the button itself as well.
  document.getElementById('resetWorkspaceBtn')?.addEventListener('click', scheduleSave);
  workspace.addEventListener('pointerup', scheduleSave, true);
  workspace.addEventListener('pointercancel', scheduleSave, true);
  workspace.addEventListener('wheel', event => {
    if (event.ctrlKey || event.metaKey) scheduleSave();
  }, { passive: true });
  document.getElementById('workspaceViewportControls')?.addEventListener('click', scheduleSave);

  window.addEventListener('pagehide', saveNow);

  async function restoreAutosaveAtStartup() {
    let saved;
    try {
      const raw = localStorage.getItem(AUTOSAVE_KEY);
      if (!raw) return;
      saved = JSON.parse(raw);
      validateProjectSnapshot(saved);
    } catch (error) {
      console.warn('자동 저장 복구 파일을 읽지 못했습니다.', error);
      return;
    }

    restoring = true;
    try {
      // runtimeVariables is intentionally empty, so graph/parameters return but
      // trained weights and other runtime state start fresh after a reload.
      await restoreProjectBeforeAutosave(saved);
      workspaceStatus.textContent = `${graph.nodes.size}개 블록 · 자동 복구됨`;
    } catch (error) {
      console.warn('작업공간 자동 복구 실패', error);
    } finally {
      restoring = false;
    }
  }

  restoreAutosaveAtStartup();
})();
