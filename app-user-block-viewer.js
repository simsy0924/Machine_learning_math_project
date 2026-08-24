// Edit user-created blocks inside the normal workspace.
//
// Opening a grouped block temporarily swaps the live workspace graph with the
// block's saved internal graph. Because the normal renderer, ports, inspector,
// pan/zoom and editing handlers are reused, the inside looks and behaves exactly
// like the rest of the app instead of being drawn by a separate viewer.

(function installUserBlockWorkspaceEditor() {
  const EXTERNAL_INPUT_TYPE = '__userBlockExternalInput';
  let editorState = null;
  let editorToolbar = null;
  let editorTitle = null;

  // Hidden source block used only while editing a user-block definition. One
  // instance represents one external input port of the collapsed custom block.
  BLOCKS[EXTERNAL_INPUT_TYPE] = {
    title: '외부 입력',
    kind: 'source',
    inputs: [],
    description: '사용자 블록 바깥에서 들어오는 입력입니다. 편집 모드에서만 보입니다.',
    formula: node => String(node.params.label || '입력'),
    controls: [{ key: 'label', label: '입력 이름', type: 'text', default: '입력' }],
    compute: () => 0
  };

  function deepClone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function ensureEditorToolbar() {
    if (editorToolbar) return;
    editorToolbar = document.createElement('div');
    editorToolbar.className = 'user-block-editor-toolbar';
    editorToolbar.hidden = true;
    editorToolbar.innerHTML = `
      <span class="pill user-block-editor-badge">사용자 블록 편집</span>
      <strong id="userBlockEditorTitle"></strong>
      <button type="button" id="userBlockEditorSave" class="primary">저장하고 돌아가기</button>
      <button type="button" id="userBlockEditorCancel" class="ghost">취소</button>`;
    document.querySelector('.workspace-head')?.appendChild(editorToolbar);
    editorTitle = document.getElementById('userBlockEditorTitle');
    document.getElementById('userBlockEditorSave')?.addEventListener('click', saveUserBlockEditor);
    document.getElementById('userBlockEditorCancel')?.addEventListener('click', cancelUserBlockEditor);
  }

  function setEditingControlsDisabled(disabled, state = editorState) {
    const ids = [
      'evaluateBtn', 'evaluateSelectedBtn', 'resetWorkspaceBtn',
      'importProjectBtn', 'exportProjectBtn', 'importWeightsBtn', 'exportWeightsBtn',
      'classifyDrawingBtn'
    ];

    if (disabled) {
      state.disabledStates = new Map();
      for (const id of ids) {
        const element = document.getElementById(id);
        if (!element) continue;
        state.disabledStates.set(id, element.disabled);
        element.disabled = true;
      }
      return;
    }

    for (const [id, wasDisabled] of state?.disabledStates || []) {
      const element = document.getElementById(id);
      if (element) element.disabled = wasDisabled;
    }
  }

  function layoutDefinition(definition) {
    const nodes = (definition.nodes || []).map(node => deepClone(node));
    if (nodes.every(node => Number.isFinite(Number(node.x)) && Number.isFinite(Number(node.y)))) {
      return nodes;
    }

    const nodeById = new Map(nodes.map(node => [Number(node.id), node]));
    const incoming = new Map(nodes.map(node => [Number(node.id), []]));
    for (const connection of definition.connections || []) {
      if (incoming.has(Number(connection.to)) && nodeById.has(Number(connection.from))) {
        incoming.get(Number(connection.to)).push(Number(connection.from));
      }
    }

    const layerMemo = new Map();
    const visiting = new Set();
    const layerOf = id => {
      if (layerMemo.has(id)) return layerMemo.get(id);
      if (visiting.has(id)) return 0;
      visiting.add(id);
      let layer = 0;
      for (const parent of incoming.get(id) || []) layer = Math.max(layer, layerOf(parent) + 1);
      visiting.delete(id);
      layerMemo.set(id, layer);
      return layer;
    };

    const layers = new Map();
    for (const node of nodes) {
      const layer = layerOf(Number(node.id));
      if (!layers.has(layer)) layers.set(layer, []);
      layers.get(layer).push(node);
    }

    const startX = 260;
    const startY = 60;
    const gapX = 245;
    const gapY = 145;
    for (const [layer, layerNodes] of layers) {
      layerNodes.forEach((node, row) => {
        node.x = startX + layer * gapX;
        node.y = startY + row * gapY;
      });
    }
    return nodes;
  }

  function buildEditorGraph(definition) {
    const internalNodes = layoutDefinition(definition);
    const nodes = new Map();
    let largestId = 0;

    for (const saved of internalNodes) {
      const id = Number(saved.id);
      if (!Number.isInteger(id) || id <= 0 || nodes.has(id)) {
        throw new Error('사용자 블록 내부 노드 ID가 올바르지 않습니다.');
      }
      getBlockDef(String(saved.type));
      const node = {
        id,
        type: String(saved.type),
        x: Number.isFinite(Number(saved.x)) ? Number(saved.x) : 260,
        y: Number.isFinite(Number(saved.y)) ? Number(saved.y) : 60,
        params: saved.params && typeof saved.params === 'object' ? deepClone(saved.params) : {}
      };
      nodes.set(id, node);
      largestId = Math.max(largestId, id);
    }

    const connections = (definition.connections || []).map(connection => ({
      from: Number(connection.from),
      to: Number(connection.to),
      inputIndex: Number(connection.inputIndex)
    })).filter(connection => nodes.has(connection.from) && nodes.has(connection.to));

    const externalInputs = Array.isArray(definition.externalInputs) ? definition.externalInputs : [];
    externalInputs.forEach((slot, index) => {
      const id = -(index + 1);
      nodes.set(id, {
        id,
        type: EXTERNAL_INPUT_TYPE,
        x: 35,
        y: 60 + index * 135,
        params: {
          label: String(slot.label || `입력 ${index + 1}`),
          externalIndex: index
        }
      });
      const targetId = Number(slot.nodeId);
      const inputIndex = Number(slot.inputIndex);
      if (nodes.has(targetId) && Number.isInteger(inputIndex) && inputIndex >= 0) {
        connections.push({ from: id, to: targetId, inputIndex });
      }
    });

    return { nodes, connections, nextId: largestId + 1 };
  }

  function decorateEditorNodes() {
    if (!editorState) return;
    nodesLayer.querySelectorAll('.node').forEach(element => {
      const id = Number(element.dataset.nodeId);
      const node = graph.nodes.get(id);
      element.classList.toggle('user-block-editor-external', node?.type === EXTERNAL_INPUT_TYPE);
      element.classList.toggle('user-block-editor-output', id === editorState.outputNodeId);
    });
  }

  function renderWorkspaceGraph() {
    nodesLayer.replaceChildren();
    wiresSvg.replaceChildren();
    for (const node of graph.nodes.values()) {
      renderNode(node);
      updateNodePreview(node);
    }
    pendingOutput = null;
    selectedNodeId = null;
    renderInspector();
    syncWorkspaceState();
    updateWires();
    decorateEditorNodes();
  }

  function openUserBlockEditor(definition) {
    if (!definition || editorState) return;
    if (evaluateBtn.disabled) {
      alert('계산이 끝난 뒤 사용자 블록을 편집해 주세요.');
      return;
    }

    cancelGroupSelection();
    cancelConnection();
    const built = buildEditorGraph(definition);
    const userBlocksSnapshot = deepClone([...USER_BLOCKS.values()]);

    editorState = {
      customId: definition.id,
      outputNodeId: Number(definition.outputNodeId),
      outerNodes: graph.nodes,
      outerConnections: graph.connections,
      outerNextNodeId: nextNodeId,
      outerSelectedNodeId: selectedNodeId,
      outerViewport: typeof getWorkspaceViewSnapshot === 'function' ? getWorkspaceViewSnapshot() : null,
      userBlocksSnapshot,
      disabledStates: new Map()
    };

    graph.nodes = built.nodes;
    graph.connections = built.connections;
    nextNodeId = built.nextId;

    ensureEditorToolbar();
    editorTitle.textContent = definition.name || '사용자 블록';
    editorToolbar.hidden = false;
    setEditingControlsDisabled(true);
    renderWorkspaceGraph();

    const savedView = definition.editorViewport;
    if (typeof restoreWorkspaceViewSnapshot === 'function') {
      restoreWorkspaceViewSnapshot(savedView || { x: 0, y: 0, zoom: 1 });
    }
    workspaceStatus.textContent = `${definition.nodes?.length || 0}개 내부 블록`;
  }

  function collectEditedDefinition() {
    if (!editorState) throw new Error('편집 중인 사용자 블록이 없습니다.');
    const current = USER_BLOCKS.get(editorState.customId);
    if (!current) throw new Error('편집 중인 사용자 블록 정의를 찾을 수 없습니다.');

    const internalNodes = [...graph.nodes.values()].filter(node => node.type !== EXTERNAL_INPUT_TYPE);
    const internalIds = new Set(internalNodes.map(node => node.id));
    if (!internalIds.has(editorState.outputNodeId)) {
      throw new Error('출력 블록이 없습니다. 내부 블록 하나를 선택해 출력으로 지정하세요.');
    }

    const externalNodes = [...graph.nodes.values()]
      .filter(node => node.type === EXTERNAL_INPUT_TYPE)
      .sort((a, b) => Number(a.params.externalIndex) - Number(b.params.externalIndex));
    const externalIds = new Set(externalNodes.map(node => node.id));

    const internalConnections = [];
    for (const connection of graph.connections) {
      if (internalIds.has(connection.from) && internalIds.has(connection.to)) {
        internalConnections.push({ ...connection });
      }
    }

    const externalInputs = [];
    for (const externalNode of externalNodes) {
      const outgoing = graph.connections.filter(connection => connection.from === externalNode.id && internalIds.has(connection.to));
      if (outgoing.length !== 1) {
        throw new Error(`외부 입력 '${externalNode.params.label || '입력'}'은 내부 입력 한 곳에 정확히 연결되어야 합니다.`);
      }
      const connection = outgoing[0];
      const target = graph.nodes.get(connection.to);
      const targetDef = getBlockDef(target.type);
      if (!Number.isInteger(connection.inputIndex) || connection.inputIndex < 0 || connection.inputIndex >= targetDef.inputs.length) {
        throw new Error('외부 입력 연결 위치가 올바르지 않습니다.');
      }
      externalInputs.push({
        nodeId: connection.to,
        inputIndex: connection.inputIndex,
        label: String(externalNode.params.label || targetDef.inputs[connection.inputIndex] || '입력')
      });
    }

    // Every internal input must be satisfied by either another internal node or
    // one of the explicit external-input source nodes.
    for (const node of internalNodes) {
      const def = getBlockDef(node.type);
      for (let inputIndex = 0; inputIndex < def.inputs.length; inputIndex++) {
        const matches = graph.connections.filter(connection => connection.to === node.id && connection.inputIndex === inputIndex);
        if (matches.length !== 1) {
          throw new Error(`'${def.title}' 블록의 '${def.inputs[inputIndex]}' 입력을 정확히 하나 연결해 주세요.`);
        }
        const sourceId = matches[0].from;
        if (!internalIds.has(sourceId) && !externalIds.has(sourceId)) {
          throw new Error('사용자 블록 내부에 올바르지 않은 연결이 있습니다.');
        }
      }
    }

    const nodes = internalNodes.map(node => ({
      id: node.id,
      type: node.type,
      x: node.x,
      y: node.y,
      params: deepClone(node.params || {})
    }));

    return {
      ...deepClone(current),
      nodes,
      connections: internalConnections,
      externalInputs,
      outputNodeId: editorState.outputNodeId,
      editorViewport: typeof getWorkspaceViewSnapshot === 'function' ? getWorkspaceViewSnapshot() : undefined
    };
  }

  function restoreOuterWorkspace(finishedState) {
    graph.nodes = finishedState.outerNodes;
    graph.connections = finishedState.outerConnections;
    nextNodeId = finishedState.outerNextNodeId;
    selectedNodeId = finishedState.outerSelectedNodeId;
    pendingOutput = null;

    nodesLayer.replaceChildren();
    wiresSvg.replaceChildren();
    for (const node of graph.nodes.values()) {
      renderNode(node);
      updateNodePreview(node);
    }

    setEditingControlsDisabled(false, finishedState);
    if (editorToolbar) editorToolbar.hidden = true;
    syncWorkspaceState();
    updateWires();
    if (typeof restoreWorkspaceViewSnapshot === 'function') {
      restoreWorkspaceViewSnapshot(finishedState.outerViewport);
    }
    renderInspector();
    renderMyBlocksPalette();
  }

  function saveUserBlockEditor() {
    if (!editorState) return;
    let updated;
    try {
      updated = collectEditedDefinition();
    } catch (error) {
      alert(`사용자 블록을 저장할 수 없습니다.\n${error.message}`);
      return;
    }

    const finishedState = editorState;
    USER_BLOCKS.set(finishedState.customId, updated); // new object invalidates compiled-plan WeakMap cache
    persistUserBlocks();
    editorState = null;
    restoreOuterWorkspace(finishedState);
    window.dispatchEvent(new CustomEvent('user-block-definition-changed', { detail: { id: updated.id } }));
  }

  function cancelUserBlockEditor() {
    if (!editorState) return;
    if (!confirm('사용자 블록 편집을 취소할까요?\n내부에서 바꾼 내용은 저장되지 않습니다.')) return;

    const finishedState = editorState;
    USER_BLOCKS.clear();
    for (const definition of finishedState.userBlocksSnapshot) USER_BLOCKS.set(definition.id, definition);
    persistUserBlocks();
    editorState = null;
    restoreOuterWorkspace(finishedState);
  }

  function setSelectedAsUserBlockOutput() {
    if (!editorState || selectedNodeId == null) return;
    const node = graph.nodes.get(selectedNodeId);
    if (!node || node.type === EXTERNAL_INPUT_TYPE) return;
    editorState.outputNodeId = node.id;
    decorateEditorNodes();
    renderInspector();
  }

  const renderInspectorBeforeUserBlockEditor = renderInspector;
  renderInspector = function() {
    renderInspectorBeforeUserBlockEditor();
    const node = selectedNodeId == null ? null : graph.nodes.get(selectedNodeId);
    if (!node || inspectorContent.hidden) return;

    if (editorState) {
      if (node.type === EXTERNAL_INPUT_TYPE) {
        const note = document.createElement('p');
        note.className = 'muted user-block-editor-input-note';
        note.textContent = '이 블록은 접힌 사용자 블록의 외부 입력 포트입니다.';
        inspectorControls.appendChild(note);
        return;
      }

      const outputButton = document.createElement('button');
      outputButton.type = 'button';
      outputButton.className = 'ghost user-block-output-button';
      outputButton.textContent = node.id === editorState.outputNodeId ? '현재 출력 블록' : '이 블록을 출력으로 지정';
      outputButton.disabled = node.id === editorState.outputNodeId;
      outputButton.addEventListener('click', setSelectedAsUserBlockOutput);
      inspectorControls.prepend(outputButton);
      return;
    }

    if (!node.type?.startsWith('custom:')) return;
    const definition = USER_BLOCKS.get(node.type.slice(7));
    if (!definition) return;

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'primary user-block-open-button';
    button.textContent = '내부 구조 편집';
    button.addEventListener('click', () => openUserBlockEditor(definition));
    inspectorControls.prepend(button);
  };

  // External-input placeholders are part of the custom block's public interface;
  // deleting one directly would silently change the number of outer ports.
  deleteNodeBtn.addEventListener('click', event => {
    if (!editorState) return;
    const node = selectedNodeId == null ? null : graph.nodes.get(selectedNodeId);
    if (node?.type !== EXTERNAL_INPUT_TYPE) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    alert('외부 입력 블록은 직접 삭제할 수 없습니다. 연결 대상과 이름은 편집할 수 있습니다.');
  }, true);

  // Never let the global workspace reset destroy the temporarily swapped graph.
  resetWorkspaceBtn.addEventListener('click', event => {
    if (!editorState) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    alert('사용자 블록 편집을 저장하거나 취소한 뒤 작업공간을 초기화해 주세요.');
  }, true);

  window.isUserBlockWorkspaceEditing = () => Boolean(editorState);
})();
