// Lightweight browser autosave for editing progress.
// Deliberately excludes RUNTIME_VARIABLES: trained weights, gradient accumulators
// and other live training state are never written by this module.

(function installWorkspaceAutosave() {
  const AUTOSAVE_KEY = 'machine-learning-math-project.workspace-autosave.v1';
  const AUTOSAVE_DELAY_MS = 220;
  let timer = null;
  let restoring = false;

  function editingUserBlockInternals() {
    return Boolean(window.isUserBlockWorkspaceEditing?.());
  }

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
    // During user-block editing graph.* intentionally points at the temporary
    // inner graph, so never let that overwrite the outer workspace autosave.
    if (restoring || editingUserBlockInternals()) return;
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
    if (restoring || editingUserBlockInternals()) return;
    if (timer != null) clearTimeout(timer);
    timer = setTimeout(saveNow, AUTOSAVE_DELAY_MS);
  }

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

  inspectorControls.addEventListener('input', scheduleSave);
  inspectorControls.addEventListener('change', scheduleSave);

  document.getElementById('classPicker')?.addEventListener('change', scheduleSave);
  document.getElementById('selectAllClassesBtn')?.addEventListener('click', scheduleSave);
  document.getElementById('clearClassSelectionBtn')?.addEventListener('click', scheduleSave);
  document.getElementById('resetWorkspaceBtn')?.addEventListener('click', scheduleSave);
  workspace.addEventListener('pointerup', scheduleSave, true);
  workspace.addEventListener('pointercancel', scheduleSave, true);
  workspace.addEventListener('wheel', event => {
    if (event.ctrlKey || event.metaKey) scheduleSave();
  }, { passive: true });
  document.getElementById('workspaceViewportControls')?.addEventListener('click', scheduleSave);

  // The editor dispatches this only after it has restored the outer graph, so the
  // next snapshot contains the normal workspace plus the newly edited definition.
  window.addEventListener('user-block-definition-changed', scheduleSave);

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
