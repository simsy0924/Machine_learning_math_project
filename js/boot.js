// Wire the buttons and start the app. Loaded last, so every function it
// references already exists and the startup order is visible in one place.

for (const button of document.querySelectorAll('[data-block]')) {
  button.addEventListener('click', () => addBlock(button.dataset.block));
}

evaluateBtn.addEventListener('click', evaluateGraph);
resetWorkspaceBtn.addEventListener('click', resetWorkspace);
deleteNodeBtn.addEventListener('click', deleteSelectedNode);
clearDrawBtn.addEventListener('click', () => {
  resetDrawCanvas();
  invalidatePreviews();
});

groupSelectBtn.addEventListener('click', startGroupSelection);
createGroupBtn.addEventListener('click', createUserBlockFromSelection);
cancelGroupBtn.addEventListener('click', cancelGroupSelection);

workspace.addEventListener('click', event => {
  if (event.target === workspace || event.target === nodesLayer || event.target === emptyState) cancelConnection();
});
// Delete / Backspace cuts the selected wire. Ignored while a form control has
// focus, where those keys still have to edit text.
window.addEventListener('keydown', event => {
  if (event.key !== 'Delete' && event.key !== 'Backspace') return;
  if (event.target?.closest?.('input, select, textarea, [contenteditable="true"]')) return;
  if (disconnectSelectedConnection()) event.preventDefault();
});

// The manual-backprop editor temporarily replaces the workspace hint. Its save
// and cancel buttons are created lazily, so use delegation and restore the normal
// hint after their own click handlers have completed. If Cancel is rejected in
// the confirmation dialog the editor is still active and the hint is kept.
document.addEventListener('click', event => {
  const id = event.target?.id;
  if (id !== 'manualBackpropEditorSave' && id !== 'manualBackpropEditorCancel') return;
  if (!window.isManualBackpropWorkspaceEditing?.()) connectionHint.textContent = DEFAULT_CONNECTION_HINT;
});

window.addEventListener('resize', updateWires);

// The workspace can change size without a window resize (for example when a
// neighboring grid panel grows). The wire SVG keeps a size-dependent viewBox,
// so redraw once after those layout changes as well. Coalescing through one
// animation frame avoids doing repeated geometry work during the same layout.
if (typeof ResizeObserver === 'function') {
  let workspaceWireResizeFrame = 0;
  const workspaceWireResizeObserver = new ResizeObserver(() => {
    if (workspaceWireResizeFrame) return;
    workspaceWireResizeFrame = requestAnimationFrame(() => {
      workspaceWireResizeFrame = 0;
      updateWires();
    });
  });
  workspaceWireResizeObserver.observe(workspace);
}

loadUserBlocks();
renderMyBlocksPalette();
resetDrawCanvas();
resetWorkspace();
restoreAutosaveAtStartup().finally(() => { document.getElementById('presentationToggle').disabled = false; });
