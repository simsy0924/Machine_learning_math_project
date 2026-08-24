// Destructive UI actions that should require explicit user intent.
// Programmatic resetWorkspace() calls (project import/autorestore) stay untouched.

(function installWorkspaceSafetyActions() {
  // Capture runs before the reset button's existing bubble listener in app-boot.
  resetWorkspaceBtn.addEventListener('click', event => {
    if (confirm('작업공간을 초기화할까요?\n현재 블록과 연결이 모두 사라집니다.')) return;
    event.preventDefault();
    event.stopImmediatePropagation();
  }, true);

  function userBlockUsage(customId) {
    const type = `custom:${customId}`;
    const graphInstances = Array.from(graph.nodes.values()).filter(node => node.type === type).length;
    const parentBlocks = [];

    for (const definition of USER_BLOCKS.values()) {
      if (definition.id === customId) continue;
      if ((definition.nodes || []).some(node => node.type === type)) {
        parentBlocks.push(definition.name || definition.id);
      }
    }

    return { graphInstances, parentBlocks };
  }

  deleteUserBlock = function(customId) {
    const definition = USER_BLOCKS.get(customId);
    if (!definition) return false;

    const usage = userBlockUsage(customId);
    if (usage.graphInstances || usage.parentBlocks.length) {
      const reasons = [];
      if (usage.graphInstances) reasons.push(`현재 작업공간에서 ${usage.graphInstances}개 사용 중`);
      if (usage.parentBlocks.length) reasons.push(`다른 사용자 블록에서 사용 중: ${usage.parentBlocks.join(', ')}`);
      alert(`'${definition.name}' 블록을 아직 삭제할 수 없습니다.\n${reasons.join('\n')}\n먼저 해당 사용처를 제거해 주세요.`);
      return false;
    }

    if (!confirm(`사용자 블록 '${definition.name}'을 삭제할까요?\n저장된 내부 구조도 함께 삭제됩니다.`)) return false;

    USER_BLOCKS.delete(customId);
    persistUserBlocks();
    renderMyBlocksPalette();
    return true;
  };

  const renderMyBlocksPaletteBeforeDeleteActions = renderMyBlocksPalette;
  renderMyBlocksPalette = function() {
    // Rebuild directly so each definition gets an add button and a separate
    // destructive action without changing the semantics of clicking the block.
    myBlocksPalette.replaceChildren();
    const definitions = [...USER_BLOCKS.values()];
    myBlocksEmpty.hidden = definitions.length > 0;

    for (const definition of definitions) {
      const row = document.createElement('div');
      row.className = 'my-block-palette-row';

      const addButton = document.createElement('button');
      addButton.className = 'palette-block custom my-block-add';
      addButton.dataset.customBlock = definition.id;
      addButton.textContent = definition.name;
      addButton.addEventListener('click', () => addBlock(`custom:${definition.id}`));

      const deleteButton = document.createElement('button');
      deleteButton.type = 'button';
      deleteButton.className = 'ghost danger my-block-delete';
      deleteButton.textContent = '삭제';
      deleteButton.title = `${definition.name} 사용자 블록 삭제`;
      deleteButton.addEventListener('click', event => {
        event.stopPropagation();
        deleteUserBlock(definition.id);
      });

      row.append(addButton, deleteButton);
      myBlocksPalette.appendChild(row);
    }
  };

  const style = document.createElement('style');
  style.textContent = `
    .my-block-palette-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 5px; align-items: stretch; margin: 4px 0; }
    .my-block-palette-row .palette-block { margin: 0; min-width: 0; }
    .my-block-delete { padding: 7px 8px; font-size: 11px; }
  `;
  document.head.appendChild(style);

  // app-boot already rendered the palette before this late-loaded module.
  renderMyBlocksPalette();
})();
