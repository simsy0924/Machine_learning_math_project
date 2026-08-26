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

// Reverse-mode gradient strategies, tried in ascending `priority` order.
// Each entry is { name, priority, plan(outputId, structureCache), execute(plan) }.
// `plan` returns null when the strategy does not apply to this graph.
const GRADIENT_STRATEGIES = [];

function registerGradientStrategy(strategy) {
  GRADIENT_STRATEGIES.push(strategy);
  GRADIENT_STRATEGIES.sort((a, b) => a.priority - b.priority);
}

// Diagnostic hooks. Every field stays null unless js/profiler.js installs it, and
// every call site checks for null first, so a page without the profiler pays only
// one property read per call.
const PROFILER_HOOKS = {
  blockCompute: null,      // { active(), record(type, node, ms) }
  kernel: null,            // { active(), record(name, ms) }
  vjp: null,               // { active(), record(type, ms) }
  gradientPass: null,      // { begin(), end(token) }
  progress: null,          // (progress) => void
  selectedRun: null,       // { begin(), end(token, result) }
  selectedLeafStep: null   // { begin() -> token|null, end(token) }
};
