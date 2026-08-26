// Lightweight browser autosave for editing progress.
//
// It listens for the one 'workspace-changed' event that every editing action
// dispatches, instead of wrapping each of those functions. Trained weights,
// gradient accumulators and other live training state are deliberately excluded:
// autosave restores what you were building, not what you had trained.

const AUTOSAVE_KEY = 'machine-learning-math-project.workspace-autosave.v1';
const AUTOSAVE_DELAY_MS = 220;

let autosaveTimer = null;
// Greater than zero while a project is being restored, so the rebuild in
// progress is never captured as a snapshot.
let autosaveSuspendDepth = 0;

function autosaveSuspended() {
  return autosaveSuspendDepth > 0 || isUserBlockWorkspaceEditing();
}

window.addEventListener('workspace-restore-start', () => { autosaveSuspendDepth++; });
window.addEventListener('workspace-restore-end', () => {
  autosaveSuspendDepth = Math.max(0, autosaveSuspendDepth - 1);
});

function buildAutosaveSnapshot() {
  return {
    ...buildProjectSnapshot({ includeRuntimeVariables: false }),
    autosave: true,
    savedAt: new Date().toISOString()
  };
}

function saveWorkspaceNow() {
  // During 사용자 블록 editing graph.* intentionally points at the temporary
  // inner graph, so never let that overwrite the outer workspace autosave.
  if (autosaveSuspended()) return;
  if (autosaveTimer != null) {
    clearTimeout(autosaveTimer);
    autosaveTimer = null;
  }
  try {
    localStorage.setItem(AUTOSAVE_KEY, JSON.stringify(buildAutosaveSnapshot()));
  } catch (error) {
    console.warn('작업공간 자동 저장 실패', error);
  }
}

function scheduleWorkspaceSave() {
  if (autosaveSuspended()) return;
  if (autosaveTimer != null) clearTimeout(autosaveTimer);
  autosaveTimer = setTimeout(saveWorkspaceNow, AUTOSAVE_DELAY_MS);
}

window.addEventListener('workspace-changed', scheduleWorkspaceSave);

// Edits that do not change the graph structure, and so do not announce
// themselves: control values, the dataset selection, and the camera.
inspectorControls.addEventListener('input', scheduleWorkspaceSave);
inspectorControls.addEventListener('change', scheduleWorkspaceSave);
classPicker?.addEventListener('change', scheduleWorkspaceSave);
selectAllClassesBtn?.addEventListener('click', scheduleWorkspaceSave);
clearClassSelectionBtn?.addEventListener('click', scheduleWorkspaceSave);
resetWorkspaceBtn?.addEventListener('click', scheduleWorkspaceSave);
workspace.addEventListener('pointerup', scheduleWorkspaceSave, true);
workspace.addEventListener('pointercancel', scheduleWorkspaceSave, true);
workspace.addEventListener('wheel', event => {
  if (event.ctrlKey || event.metaKey) scheduleWorkspaceSave();
}, { passive: true });
document.getElementById('workspaceViewportControls')?.addEventListener('click', scheduleWorkspaceSave);

window.addEventListener('pagehide', saveWorkspaceNow);

// Restoring is itself a series of workspace changes; suspending autosave keeps
// the stored snapshot from being rewritten with what it just produced.
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

  // Held across restoreProject's own suspension so the notify it ends with does
  // not immediately rewrite the snapshot we just read.
  autosaveSuspendDepth++;
  try {
    await restoreProject(saved);
    workspaceStatus.textContent = `${graph.nodes.size}개 블록 · 자동 복구됨`;
  } catch (error) {
    console.warn('작업공간 자동 복구 실패', error);
  } finally {
    autosaveSuspendDepth = Math.max(0, autosaveSuspendDepth - 1);
  }
}
