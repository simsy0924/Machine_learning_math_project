// Safety rules for editing a grouped block in the normal workspace.
// These keep the editor permissive for ordinary math while preventing structures
// the user-block runtime cannot execute as a pure reusable function.

(function installUserBlockEditorGuards() {
  const EXTERNAL_INPUT_TYPE = '__userBlockExternalInput';
  const UNSUPPORTED_IN_USER_BLOCK = new Set(['derivative', 'setVariable', 'repeat']);

  // Grouping an external-input placeholder would persist an editor-only block type
  // inside another custom definition. Keep those interface nodes outside groups.
  const toggleGroupNodeBeforeUserBlockGuard = toggleGroupNode;
  toggleGroupNode = function(id) {
    if (window.isUserBlockWorkspaceEditing?.() && graph.nodes.get(id)?.type === EXTERNAL_INPUT_TYPE) {
      alert('외부 입력 블록은 사용자 블록의 입력 포트이므로 다른 블록과 다시 묶을 수 없습니다.');
      return;
    }
    return toggleGroupNodeBeforeUserBlockGuard(id);
  };

  // app-boot installed palette click listeners before this late-loaded module.
  // Capture at document level so unsupported stateful/special blocks never reach
  // their ordinary addBlock listeners while a custom definition is open.
  document.addEventListener('click', event => {
    if (!window.isUserBlockWorkspaceEditing?.()) return;
    const button = event.target.closest('[data-block]');
    if (!button) return;
    const type = String(button.dataset.block || '');
    if (!UNSUPPORTED_IN_USER_BLOCK.has(type)) return;

    event.preventDefault();
    event.stopImmediatePropagation();
    alert('미분, 값 바꾸기, 반복 블록은 사용자 블록 내부에 넣을 수 없습니다. 사용자 블록 바깥에서 사용해 주세요.');
  }, true);

  // Also validate before saving, protecting imported/legacy or programmatically
  // constructed editor graphs even if they somehow contain a forbidden block.
  document.addEventListener('click', event => {
    if (!window.isUserBlockWorkspaceEditing?.()) return;
    if (!event.target.closest('#userBlockEditorSave')) return;

    const forbidden = [...graph.nodes.values()].find(node => UNSUPPORTED_IN_USER_BLOCK.has(node.type));
    if (!forbidden) return;

    event.preventDefault();
    event.stopImmediatePropagation();
    const title = (() => {
      try { return getBlockDef(forbidden.type).title; }
      catch { return forbidden.type; }
    })();
    alert(`'${title}' 블록은 사용자 블록 내부에 저장할 수 없습니다. 제거한 뒤 다시 저장해 주세요.`);
  }, true);
})();
