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
window.addEventListener('resize', updateWires);

loadUserBlocks();
renderMyBlocksPalette();
resetDrawCanvas();
resetWorkspace();
restoreAutosaveAtStartup();
