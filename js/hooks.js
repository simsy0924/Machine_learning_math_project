// Extension points.
//
// Earlier versions of this app extended existing behaviour by reassigning the
// global function that already existed (`evaluateNode = function ...`). With a
// dozen files doing that, the code you read was rarely the code that ran. The
// rule now is: a function is defined exactly once, in the file that owns it, and
// anything that wants to observe or extend it registers here instead.

// Called after the inspector has rendered the built-in controls for a node.
// Signature: (node, def) => void
const INSPECTOR_EXTENSIONS = [];

// Called after "선택 계산" has rendered its result panel. Signature: (value) => void
const SELECTED_RESULT_EXTENSIONS = [];

// Kept as an empty compatibility surface for older diagnostics. Automatic
// differentiation no longer registers or executes gradient strategies.
const GRADIENT_STRATEGIES = [];

function registerGradientStrategy(strategy) {
  GRADIENT_STRATEGIES.push(strategy);
  GRADIENT_STRATEGIES.sort((a, b) => a.priority - b.priority);
}

// Diagnostic hooks. Every field stays null unless js/profiler.js installs it, and
// every call site checks for null first, so a page without the profiler pays only
// a tiny branch on the hot path.
const PROFILER_HOOKS = {
  blockCompute: null, // { active(), record(type, node, ms) } for interpreted/selected paths
  kernel: null,       // { active(), record(name, ms) }
  progress: null,     // (progress) => void
  selectedRun: null,  // { begin(), end(token, result), finish(token) }
  leafStep: null,     // { begin(mode) -> token|null, end(token) } around one compiled repeat iteration
  leafNode: null      // { active(mode), record(type, node, ms) } around each compiled plan step
};