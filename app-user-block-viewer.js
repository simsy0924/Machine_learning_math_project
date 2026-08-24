// Read-only structure viewer for user-created blocks.
// A grouped block is an abstraction, not a black box: this viewer reconstructs
// the saved internal graph without modifying the live workspace.

(function installUserBlockStructureViewer() {
  let viewer = null;
  let viewerTitle = null;
  let viewerBreadcrumb = null;
  let viewerCanvas = null;
  let viewerStack = [];

  function ensureViewer() {
    if (viewer) return;

    viewer = document.createElement('div');
    viewer.className = 'user-block-viewer';
    viewer.hidden = true;
    viewer.innerHTML = `
      <div class="user-block-viewer-backdrop" data-viewer-close></div>
      <section class="user-block-viewer-dialog" role="dialog" aria-modal="true" aria-label="사용자 블록 내부 구조">
        <header class="user-block-viewer-head">
          <div>
            <p class="eyebrow">Grouped math block</p>
            <h2 id="userBlockViewerTitle">내부 구조</h2>
            <div id="userBlockViewerBreadcrumb" class="user-block-viewer-breadcrumb"></div>
          </div>
          <div class="user-block-viewer-actions">
            <button type="button" id="userBlockViewerBack" class="ghost" hidden>이전 블록</button>
            <button type="button" class="ghost" data-viewer-close>닫기</button>
          </div>
        </header>
        <div class="user-block-viewer-note">읽기 전용 · 묶기 전의 수학 블록과 연결을 그대로 보여줍니다.</div>
        <div id="userBlockViewerCanvas" class="user-block-viewer-canvas"></div>
      </section>`;
    document.body.appendChild(viewer);

    viewerTitle = document.getElementById('userBlockViewerTitle');
    viewerBreadcrumb = document.getElementById('userBlockViewerBreadcrumb');
    viewerCanvas = document.getElementById('userBlockViewerCanvas');

    viewer.querySelectorAll('[data-viewer-close]').forEach(button => button.addEventListener('click', closeUserBlockViewer));
    document.getElementById('userBlockViewerBack').addEventListener('click', () => {
      if (viewerStack.length <= 1) return;
      viewerStack.pop();
      renderViewerDefinition(viewerStack[viewerStack.length - 1]);
    });
  }

  function nodeKindLabel(kind) {
    return ({
      source: '입력', data: '데이터', generator: '생성', transform: '변환',
      operation: '연산', calculus: '미분', custom: '내 블록', sink: '확인',
      state: '상태', control: '제어'
    })[kind] || kind || '블록';
  }

  function buildDefinitionLayout(definition) {
    const nodes = Array.isArray(definition.nodes) ? definition.nodes : [];
    const connections = Array.isArray(definition.connections) ? definition.connections : [];
    const externalInputs = Array.isArray(definition.externalInputs) ? definition.externalInputs : [];
    const nodeById = new Map(nodes.map(node => [node.id, node]));
    const incoming = new Map(nodes.map(node => [node.id, []]));

    for (const connection of connections) {
      if (incoming.has(connection.to) && nodeById.has(connection.from)) incoming.get(connection.to).push(connection.from);
    }

    const layerMemo = new Map();
    const visiting = new Set();
    const layerOf = id => {
      if (layerMemo.has(id)) return layerMemo.get(id);
      if (visiting.has(id)) return 0;
      visiting.add(id);
      const parents = incoming.get(id) || [];
      let layer = 0;
      for (const parent of parents) layer = Math.max(layer, layerOf(parent) + 1);
      visiting.delete(id);
      layerMemo.set(id, layer);
      return layer;
    };

    const layers = new Map();
    for (const node of nodes) {
      const layer = layerOf(node.id);
      if (!layers.has(layer)) layers.set(layer, []);
      layers.get(layer).push(node);
    }

    const cardW = 174;
    const cardH = 84;
    const gapX = 86;
    const gapY = 38;
    const externalW = 142;
    const leftPad = 34;
    const topPad = 44;
    const externalX = leftPad;
    const internalStartX = externalInputs.length ? leftPad + externalW + 94 : leftPad;
    const positions = new Map();

    const sortedLayers = [...layers.keys()].sort((a, b) => a - b);
    let maxRows = Math.max(1, externalInputs.length);
    for (const layer of sortedLayers) maxRows = Math.max(maxRows, layers.get(layer).length);

    for (const layer of sortedLayers) {
      const layerNodes = layers.get(layer);
      const totalHeight = layerNodes.length * cardH + Math.max(0, layerNodes.length - 1) * gapY;
      const availableHeight = maxRows * cardH + Math.max(0, maxRows - 1) * gapY;
      const startY = topPad + Math.max(0, (availableHeight - totalHeight) / 2);
      layerNodes.forEach((node, row) => {
        positions.set(node.id, {
          x: internalStartX + layer * (cardW + gapX),
          y: startY + row * (cardH + gapY),
          w: cardW,
          h: cardH
        });
      });
    }

    const externalPositions = externalInputs.map((slot, index) => ({
      x: externalX,
      y: topPad + index * (cardH + gapY),
      w: externalW,
      h: 60,
      slot,
      index
    }));

    const maxLayer = sortedLayers.length ? Math.max(...sortedLayers) : 0;
    const width = internalStartX + (maxLayer + 1) * cardW + maxLayer * gapX + 70;
    const height = topPad * 2 + maxRows * cardH + Math.max(0, maxRows - 1) * gapY;
    return { nodes, connections, externalInputs, positions, externalPositions, width: Math.max(560, width), height: Math.max(320, height) };
  }

  function makePath(x1, y1, x2, y2) {
    const bend = Math.max(45, Math.abs(x2 - x1) * 0.42);
    return `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`;
  }

  function renderViewerDefinition(definition) {
    ensureViewer();
    const layout = buildDefinitionLayout(definition);
    viewerTitle.textContent = definition.name || '사용자 블록';
    viewerBreadcrumb.textContent = viewerStack.map(item => item.name || '사용자 블록').join('  ›  ');
    document.getElementById('userBlockViewerBack').hidden = viewerStack.length <= 1;
    viewerCanvas.replaceChildren();

    const stage = document.createElement('div');
    stage.className = 'user-block-viewer-stage';
    stage.style.width = `${layout.width}px`;
    stage.style.height = `${layout.height}px`;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.classList.add('user-block-viewer-wires');
    svg.setAttribute('viewBox', `0 0 ${layout.width} ${layout.height}`);
    stage.appendChild(svg);

    const internalCardById = new Map();
    for (const node of layout.nodes) {
      const pos = layout.positions.get(node.id);
      if (!pos) continue;
      let def;
      try { def = getBlockDef(node.type); }
      catch { def = { title: node.type, kind: 'operation', formula: () => node.type }; }

      const card = document.createElement('div');
      card.className = `user-block-viewer-node ${def.kind || ''}`;
      if (node.id === definition.outputNodeId) card.classList.add('output-node');
      if (node.type.startsWith('custom:')) card.classList.add('nested-custom');
      card.style.left = `${pos.x}px`;
      card.style.top = `${pos.y}px`;
      card.style.width = `${pos.w}px`;
      card.dataset.internalNodeId = node.id;
      let formula = '';
      try { formula = def.formula ? def.formula(node) : ''; } catch {}
      card.innerHTML = `
        <div class="user-block-viewer-node-head"><strong>${escapeHtml(def.title || node.type)}</strong><span>${escapeHtml(nodeKindLabel(def.kind))}</span></div>
        <div class="user-block-viewer-node-formula">${escapeHtml(formula || '')}</div>
        ${node.id === definition.outputNodeId ? '<div class="user-block-viewer-output-label">출력</div>' : ''}`;

      if (node.type.startsWith('custom:')) {
        const nestedId = node.type.slice(7);
        const nestedDefinition = USER_BLOCKS.get(nestedId);
        if (nestedDefinition) {
          card.title = '눌러서 이 사용자 블록의 내부 보기';
          card.addEventListener('click', () => {
            viewerStack.push(nestedDefinition);
            renderViewerDefinition(nestedDefinition);
          });
        }
      }

      stage.appendChild(card);
      internalCardById.set(node.id, card);
    }

    for (const ext of layout.externalPositions) {
      const card = document.createElement('div');
      card.className = 'user-block-viewer-external';
      card.style.left = `${ext.x}px`;
      card.style.top = `${ext.y}px`;
      card.style.width = `${ext.w}px`;
      card.innerHTML = `<span>외부 입력 ${ext.index + 1}</span><strong>${escapeHtml(ext.slot.label || `입력 ${ext.index + 1}`)}</strong>`;
      stage.appendChild(card);
    }

    // Internal wires.
    for (const connection of layout.connections) {
      const from = layout.positions.get(connection.from);
      const to = layout.positions.get(connection.to);
      if (!from || !to) continue;
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', makePath(from.x + from.w, from.y + from.h / 2, to.x, to.y + to.h / 2));
      path.setAttribute('class', 'user-block-viewer-wire');
      svg.appendChild(path);
    }

    // External-input wires. Each saved slot points to the internal input it used
    // to feed before the group was collapsed.
    layout.externalPositions.forEach(ext => {
      const target = layout.positions.get(ext.slot.nodeId);
      if (!target) return;
      const yOffset = ext.slot.inputIndex == null ? target.h / 2 : Math.min(target.h - 14, 24 + Number(ext.slot.inputIndex) * 17);
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', makePath(ext.x + ext.w, ext.y + ext.h / 2, target.x, target.y + yOffset));
      path.setAttribute('class', 'user-block-viewer-wire external-wire');
      svg.appendChild(path);
    });

    viewerCanvas.appendChild(stage);
  }

  function openUserBlockViewer(definition) {
    if (!definition) return;
    ensureViewer();
    viewerStack = [definition];
    renderViewerDefinition(definition);
    viewer.hidden = false;
    document.body.classList.add('user-block-viewer-open');
  }

  function closeUserBlockViewer() {
    if (!viewer) return;
    viewer.hidden = true;
    viewerStack = [];
    document.body.classList.remove('user-block-viewer-open');
  }

  const renderInspectorBeforeUserBlockViewer = renderInspector;
  renderInspector = function() {
    renderInspectorBeforeUserBlockViewer();
    const node = selectedNodeId == null ? null : graph.nodes.get(selectedNodeId);
    if (!node?.type?.startsWith('custom:') || inspectorContent.hidden) return;

    const definition = USER_BLOCKS.get(node.type.slice(7));
    if (!definition) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'primary user-block-open-button';
    button.textContent = '내부 구조 보기';
    button.addEventListener('click', () => openUserBlockViewer(definition));
    inspectorControls.prepend(button);
  };

  window.addEventListener('keydown', event => {
    if (event.key === 'Escape' && viewer && !viewer.hidden) closeUserBlockViewer();
  });
})();
