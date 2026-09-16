// Run with: node bench/presentation-check.mjs
// Exercise real file IO, autosave and presentation transitions with a small DOM
// adapter. This does not replace visual browser QA.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, { id, disabled:false, hidden:false, value:'', innerHTML:'', textContent:'',
    dataset:{},classList:{add(){},toggle(){}},addEventListener(){}, setAttribute(){},
    querySelectorAll(){return []}, querySelector(){return null}, replaceChildren(){}, appendChild(){}, click(){}, remove(){} });
  return elements.get(id);
}
const storage = new Map(); const window = new EventTarget();
let files = new Map(); let requests = []; let blocked = null;
const context = vm.createContext({ console,window,Map,Set,Float32Array,Uint8Array,URL,Blob,AbortSignal,Event,
  structuredClone, setTimeout,clearTimeout, btoa,atob,
  CustomEvent: class extends Event {}, Option:class {}, alert:message=>{throw Error(message)},
  localStorage:{getItem:key=>storage.get(key)??null,setItem:(key,value)=>storage.set(key,value)},
  document:{baseURI:'https://example.test/app/',getElementById:element,querySelector:element,
    querySelectorAll:()=>[],createElement:()=>element('download'),body:element('body')},
  fetch:async (url,options)=>{
    requests.push({url:String(url),cache:options.cache});
    if(blocked) await blocked;
    const data=files.get(new URL(url).pathname);
    return {ok:data!==undefined,status:data===undefined?404:200,headers:{get:()=>null},blob:async()=>new Blob([JSON.stringify(data)])};
  }
});
vm.runInContext(`
let presentationActive=false,presentationBusy=false;
const graph={nodes:new Map(),connections:[]}, USER_BLOCKS=new Map(),RUNTIME_VARIABLES=new Map(),datasetCache=new Map();
let nextNodeId=1,selectedNodeId=null,pendingOutput=null,selectedConnection=null,groupSelectionMode=false;
const groupSelectedIds=new Set(); let datasetReviewRun=null,datasetReviewPosition=0,datasetReviewBusy=false,datasetReviewStop=false;
const rawStrokes=[];let activeStroke=null,drawingPointerDown=false;
let view={x:0,y:0,zoom:1}, classes=[];
const drawCanvas={width:2,height:2},previewCanvas=drawCanvas;
const drawCtx={getImageData:()=>({pixels:[1,2]}),putImageData(){}},previewCtx=drawCtx;
const nodesLayer=document.getElementById('nodes'),wiresSvg=document.getElementById('wires');
const workspace=document.getElementById('workspace'),inspectorControls=document.getElementById('inspectorControls');
const evaluateBtn=document.getElementById('evaluateBtn'),evaluateSelectedBtn=document.getElementById('evaluateSelectedBtn');
const classPicker=document.getElementById('classPicker'),loadDatasetBtn=document.getElementById('loadDatasetBtn');
const selectAllClassesBtn=document.getElementById('selectAllClassesBtn'),clearClassSelectionBtn=document.getElementById('clearClassSelectionBtn');
const resetWorkspaceBtn=document.getElementById('resetWorkspaceBtn'),workspaceStatus=document.getElementById('workspaceStatus');
const USER_BLOCK_STORAGE_KEY='user-blocks';
function isUserBlockWorkspaceEditing(){return false}
function bumpUserBlockLibraryVersion(){}
function getWorkspaceViewSnapshot(){return {...view}}
function restoreWorkspaceViewSnapshot(v){view=v?{...v}:{x:0,y:0,zoom:1}}
function notifyWorkspaceChanged(){window.dispatchEvent(new Event('workspace-changed'))}
function resetWorkspace(){graph.nodes.clear();graph.connections=[];RUNTIME_VARIABLES.clear();notifyWorkspaceChanged()}
function getBlockDef(type){if(!['number','variable'].includes(type))throw Error('unknown block');return {}}
function renderNode(){} function renderMyBlocksPalette(){} function renderInspector(){}
function syncWorkspaceState(){} function updateWires(){} function invalidatePreviews(){}
function updateDatasetSummary(){} function renderRandomSamples(){} function updateNodePreview(){}
function renderDatasetReviewImage(){} function startGroupSelection(){groupSelectionMode=true} function updateGroupUI(){}
function resetDrawCanvas(){rawStrokes.length=0}
function isArrayValue(v){return v?.data instanceof Float32Array}
function asArrayValue(v){return v} function arrayValue(data,shape){return {data,shape}}
function copyValue(v){return structuredClone(v)}
window.quickDrawDataset={selectedClassNames:()=>classes,loadSelectedClasses:async()=>{for(const name of classes)datasetCache.set(name,{name})}};
`,context);
for(const file of ['js/file-io.js','js/autosave.js','js/presentation.js'])vm.runInContext(fs.readFileSync(new URL('../'+file,import.meta.url),'utf8'),context,{filename:file});
// Use the production custom-block persistence function as well.
const userBlocks=fs.readFileSync(new URL('../js/user-blocks.js',import.meta.url),'utf8');
vm.runInContext(userBlocks.slice(userBlocks.indexOf('function persistUserBlocks()'),userBlocks.indexOf('function loadUserBlocks()')),context);
const run=code=>vm.runInContext(code,context);
const snapshot=()=>JSON.parse(run('JSON.stringify(buildProjectSnapshot())'));
const comparable=project=>{delete project.exportedAt;return project};
const blank={format:'machine-learning-math-project',version:1,graph:{nodes:[],connections:[]},userBlocks:[],runtimeVariables:[],datasetSelection:[]};
files.set('/app/presentation/steps.json',{steps:[{id:'01',file:'01.mmlab'},{id:'02',file:'02.mmlab'}]});
files.set('/app/presentation/descriptions.json',{'01':'첫째 줄\n둘째 줄'});
files.set('/app/presentation/01.mmlab',blank);files.set('/app/presentation/02.mmlab',blank);
run(`graph.nodes.set(7,{id:7,type:'number',x:20,y:40,params:{value:42}});selectedNodeId=7;
RUNTIME_VARIABLES.set('W',{signature:'weight',value:arrayValue(new Float32Array([1.25,2.5]),[2])});
USER_BLOCKS.set('editor',{id:'editor',name:'편집 블록'});persistUserBlocks();saveWorkspaceNow();
datasetCache.set('cat',{name:'cat',bytes:new Uint8Array([3])});rawStrokes.push([{x:1,y:2}]);view={x:14,y:28,zoom:1.5};`);
const original=comparable(snapshot());
await run('changePresentation(0)');
const stored=new Map(storage);
assert.equal(run('presentationActive'),true);assert.equal(run('graph.nodes.size'),0);
assert.equal(elements.get('presentationDescriptionText').textContent,'첫째 줄\n둘째 줄');
run(`graph.nodes.set(2,{id:2,type:'number',params:{value:99}});RUNTIME_VARIABLES.set('W',{signature:'stage',value:123});USER_BLOCKS.clear();persistUserBlocks();saveWorkspaceNow();`);
assert.deepEqual(storage,stored);
await run('changePresentation(1)');await run('changePresentation(0)');
assert.equal(run('graph.nodes.size'),0);assert.equal(run('RUNTIME_VARIABLES.size'),0);
// Missing file must leave the current stage untouched.
files.delete('/app/presentation/02.mmlab');await run('changePresentation(1)');
assert.equal(run('presentationIndex'),0);assert.match(elements.get('presentationStatus').textContent,/404/);
// Invalid node fails AFTER clearing the workspace: exercise rollback.
files.set('/app/presentation/02.mmlab',{...blank,graph:{nodes:[{id:1,type:'broken'}],connections:[]}});
run(`graph.nodes.set(3,{id:3,type:'number',params:{value:88}})`);await run('changePresentation(1)');
assert.equal(run('graph.nodes.get(3).params.value'),88);assert.equal(run('presentationIndex'),0);
await run("changePresentation('off')");assert.equal(run('presentationActive'),false);
assert.deepEqual(comparable(snapshot()),original);assert.deepEqual(storage,stored);
assert.equal(run('datasetCache.get("cat").bytes[0]'),3);assert.equal(run('rawStrokes[0][0].x'),1);assert.equal(run('selectedNodeId'),7);
// Changed server file is re-fetched on re-entry, and rapid next cannot race it.
files.set('/app/presentation/01.mmlab',{...blank,graph:{nodes:[{id:5,type:'number',params:{value:55}}],connections:[]}});
let release;blocked=new Promise(resolve=>release=resolve);
const entering=run('changePresentation(0)');await run('changePresentation(1)');assert.equal(run('presentationBusy'),true);
release();blocked=null;await entering;assert.equal(run('graph.nodes.get(5).params.value'),55);
assert(requests.every(r=>r.cache==='no-store'));
run('evaluateBtn.disabled=true');await run('changePresentation(1)');assert.equal(run('presentationIndex'),0);run('evaluateBtn.disabled=false');
await run("changePresentation('off')");assert.deepEqual(comparable(snapshot()),original);
console.log('PASS: editor graph/Float32 weights/custom blocks/storage/data/drawing/view restoration; independent stages; fresh GETs; rollback; transition race and running-computation guards.');
