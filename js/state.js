// DOM handles, the Quick Draw class table, and the mutable state every other
// module reads. Nothing here computes anything; it only names the things that
// exist for the whole life of the page.

const workspace = document.getElementById('workspace');
const nodesLayer = document.getElementById('nodes');
const wiresSvg = document.getElementById('wires');
const emptyState = document.getElementById('emptyState');
const workspaceStatus = document.getElementById('workspaceStatus');
const connectionHint = document.getElementById('connectionHint');
const evaluateBtn = document.getElementById('evaluateBtn');
const evaluateSelectedBtn = document.getElementById('evaluateSelectedBtn');
const resetWorkspaceBtn = document.getElementById('resetWorkspaceBtn');

const inspectorEmpty = document.getElementById('inspectorEmpty');
const inspectorContent = document.getElementById('inspectorContent');
const inspectorTitle = document.getElementById('inspectorTitle');
const inspectorDescription = document.getElementById('inspectorDescription');
const inspectorFormula = document.getElementById('inspectorFormula');
const inspectorValue = document.getElementById('inspectorValue');
const inspectorControls = document.getElementById('inspectorControls');
const inspectorConnections = document.getElementById('inspectorConnections');
const deleteNodeBtn = document.getElementById('deleteNodeBtn');

const groupSelectBtn = document.getElementById('groupSelectBtn');
const createGroupBtn = document.getElementById('createGroupBtn');
const cancelGroupBtn = document.getElementById('cancelGroupBtn');
const myBlocksPalette = document.getElementById('myBlocksPalette');
const myBlocksEmpty = document.getElementById('myBlocksEmpty');

const drawCanvas = document.getElementById('drawCanvas');
const drawCtx = drawCanvas.getContext('2d', { willReadFrequently: true });
const previewCanvas = document.getElementById('previewCanvas');
const previewCtx = previewCanvas.getContext('2d');
const clearDrawBtn = document.getElementById('clearDrawBtn');

// The one list of Quick Draw classes. The block palette, the dataset loader and
// the result panels all read their names and Korean labels from here.
const QUICK_DRAW_CLASSES = [
  ['cat', '고양이'], ['fish', '물고기'], ['house', '집'], ['tree', '나무'],
  ['car', '자동차'], ['apple', '사과'], ['clock', '시계'], ['star', '별'],
  ['umbrella', '우산'], ['airplane', '비행기'], ['face', '얼굴'], ['flower', '꽃'],
  ['cup', '컵'], ['bicycle', '자전거'], ['guitar', '기타']
];

function classOptions() {
  return QUICK_DRAW_CLASSES.map(([value, label]) => ({ value, label }));
}

function koreanClassName(name) {
  const found = QUICK_DRAW_CLASSES.find(([value]) => value === name);
  return found ? found[1] : name;
}

// ---------- workspace state ----------

let nextNodeId = 1;
let selectedNodeId = null;
let pendingOutput = null;
// The wire the user picked, as { to, inputIndex }. An input port holds at most
// one connection, so that pair names one wire; keeping the pair instead of the
// connection object survives the array being rebuilt on every edit.
let selectedConnection = null;
let groupSelectionMode = false;
const groupSelectedIds = new Set();

const graph = { nodes: new Map(), connections: [] };
const USER_BLOCKS = new Map();
const USER_BLOCK_STORAGE_KEY = 'machine-learning-math-project.user-blocks.v1';

// Compiled plans capture the 사용자 블록 definitions they saw while compiling —
// including the definitions of nested 사용자 블록. Keying those caches on object
// identity alone is not enough: editing a nested block replaces only that one
// object, and every parent plan keeps running the version it captured. Every
// change to the library bumps this counter instead, and a plan compiled at an
// older version is rebuilt on its next use.
let USER_BLOCK_LIBRARY_VERSION = 0;

function bumpUserBlockLibraryVersion() {
  USER_BLOCK_LIBRARY_VERSION++;
}

// Block type used only while a 사용자 블록 definition is open in the workspace
// editor. One node of this type stands for one external input port.
const USER_BLOCK_EXTERNAL_INPUT_TYPE = '__userBlockExternalInput';

// ---------- training runtime state ----------

// Values of 변수 blocks that 값 바꾸기 has written. Inside a 반복 the writes are
// buffered in pendingVariableUpdates and committed at the end of the iteration.
const RUNTIME_VARIABLES = new Map();
let pendingVariableUpdates = null;

// ---------- change notification ----------

// Anything that edits the workspace announces it once instead of being wrapped
// by the modules that care (autosave listens for this).
function notifyWorkspaceChanged() {
  window.dispatchEvent(new CustomEvent('workspace-changed'));
}
