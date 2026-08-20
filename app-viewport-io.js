// Keep the workspace camera position in .mmlab files without embedding any dataset bytes.

const buildProjectSnapshotWithoutViewport = buildProjectSnapshot;
buildProjectSnapshot = function() {
  const snapshot = buildProjectSnapshotWithoutViewport();
  snapshot.viewport = getWorkspaceViewSnapshot();
  return snapshot;
};

const restoreProjectWithoutViewport = restoreProject;
restoreProject = async function(project) {
  await restoreProjectWithoutViewport(project);
  restoreWorkspaceViewSnapshot(project?.viewport);
};
