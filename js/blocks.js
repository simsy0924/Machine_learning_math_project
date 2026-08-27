// The block catalogue: every block the palette can create, defined exactly once.
//
// A block definition is:
//   title, kind, inputs[], description, formula(node), controls[]
//   compute(node, inputs) -> value
//   vjp(inputs, output, upstream) -> gradient per input   (optional)
//   special: 'repeat' | 'setVariable' | 'derivative'      (optional)
//
// `special` blocks are not ordinary functions of their inputs — the evaluator in
// js/evaluate.js drives them. Everything else is pure and its derivative lives
// right next to its forward pass, so adding a block never means editing an
// autodiff switch statement somewhere else.

// Caches for source blocks whose value depends only on their own controls.
// Keyed by the node object, so editing a control rebuilds and moving a node does
// not. Their results outlive an iteration, so they never use the result arena.
const CONSTANT_VECTOR_CACHE = new WeakMap();
const RANDOM_VECTOR_CACHE = new WeakMap();
const MATRIX_CACHE = new WeakMap();
const ONE_HOT_CACHE = new Map();

let cachedDatasetClasses = [];
let cachedDatasetValue = null;

// Set while 손그림 테스트 runs, so the model reads the hand drawing wherever the
// graph reads 그림값. Cleared again as soon as that evaluation finishes.
let handDrawingSampleOverride = null;

function setHandDrawingSampleOverride(produceValue) {
  handDrawingSampleOverride = produceValue;
}

function cachedNodeValue(cache, node, signature, build) {
  const previous = cache.get(node);
  if (previous && previous.signature === signature) return previous.value;
  const value = build();
  cache.set(node, { signature, value });
  return value;
}

function currentDatasetValue() {
  const api = window.quickDrawDataset;
  if (!api) throw new Error('데이터 로더가 아직 준비되지 않았습니다.');
  const cache = api.cache;
  if (!cache || !cache.size) throw new Error('오른쪽에서 Quick Draw 데이터를 먼저 불러오세요.');

  let unchanged = cache.size === cachedDatasetClasses.length;
  if (unchanged) {
    let i = 0;
    for (const name of cache.keys()) {
      if (cachedDatasetClasses[i++] !== name) {
        unchanged = false;
        break;
      }
    }
  }

  if (!unchanged || !cachedDatasetValue) {
    cachedDatasetClasses = Array.from(cache.keys());
    cachedDatasetValue = { kind: 'dataset', classes: cachedDatasetClasses };
  }
  return cachedDatasetValue;
}

// Keep the UI permissive for large experiments while still having a finite cap.
const MAX_REPEAT_COUNT = 1_000_000;

function repeatCount(node) {
  return Math.max(0, Math.min(MAX_REPEAT_COUNT, Math.floor(Number(node?.params?.count) || 0)));
}

function repeatStartValue(node) {
  return Math.floor(Number(node?.params?.start) || 0);
}

function op1(title, formula, fn, vjp) {
  return {
    title, kind: 'operation', inputs: ['x'],
    description: `${title} 연산을 적용한다.`,
    formula: () => formula,
    compute: (n, [x]) => fn(x),
    vjp
  };
}

function op2(title, formula, fn, vjp) {
  return {
    title, kind: 'operation', inputs: ['a', 'b'],
    description: `${title} 연산을 적용한다. 숫자와 배열의 조합도 지원한다.`,
    formula: () => formula,
    compute: (n, [a, b]) => fn(a, b),
    vjp
  };
}

const BLOCKS = {
  // ---------- 입력 ----------
  number: {
    title: '숫자', kind: 'source', inputs: [], description: '하나의 실수 값을 만든다.',
    formula: n => String(n.params.value),
    controls: [{ key: 'value', label: '값', type: 'number', step: 'any', default: 1 }],
    compute: n => Number(n.params.value)
  },

  variable: {
    title: '변수', kind: 'source', inputs: [],
    description: '이름과 값을 가진 변수. 미분할 대상을 지정할 때 사용한다.',
    formula: node => {
      if (node.params.mode === 'matrix') return `${node.params.name} ∈ R^(${node.params.rows}×${node.params.cols})`;
      if (node.params.mode === 'vector') return `${node.params.name} ∈ R^${node.params.length}`;
      return `${node.params.name} = ${node.params.value}`;
    },
    controls: [
      { key: 'name', label: '변수 이름', type: 'text', default: 'x' },
      {
        key: 'mode', label: '형태', type: 'select', default: 'scalar',
        options: [{ value: 'scalar', label: '숫자' }, { value: 'vector', label: '벡터' }, { value: 'matrix', label: '행렬' }]
      },
      { key: 'value', label: '숫자 값 / 벡터 채움값', type: 'number', step: 'any', default: 1 },
      { key: 'length', label: '벡터 길이', type: 'number', step: '1', default: 3 },
      { key: 'init', label: '벡터 초기값', type: 'select', default: 'constant', options: [{ value: 'constant', label: '같은 값' }, { value: 'random', label: '무작위(-1~1)' }] },
      { key: 'seed', label: '무작위 시드', type: 'number', step: '1', default: 1 },
      { key: 'rows', label: '행렬 행 수', type: 'number', step: '1', default: 3 },
      { key: 'cols', label: '행렬 열 수', type: 'number', step: '1', default: 3 },
      { key: 'scale', label: '무작위 범위', type: 'number', step: 'any', default: 0.05 }
    ],
    compute: node => variableBlockValue(node)
  },

  drawing: {
    title: '그림 입력', kind: 'source', inputs: [], description: '오른쪽 그림판의 현재 그림을 입력으로 사용한다.',
    formula: () => 'image', compute: () => ({ kind: 'image', canvas: drawCanvas })
  },

  // ---------- 데이터 ----------
  dataset: {
    title: '불러온 데이터', kind: 'data', inputs: [], description: '오른쪽에서 현재 불러온 Quick Draw 종류들을 가져온다.',
    formula: () => 'D',
    compute: () => currentDatasetValue()
  },

  chooseClass: {
    title: '종류 고르기', kind: 'data', inputs: ['데이터'], description: '불러온 데이터 중 한 종류를 고른다.',
    formula: n => `class = ${n.params.className}`,
    controls: [{ key: 'className', label: '종류', type: 'select', default: 'cat', options: classOptions() }],
    compute: (n, [dataset]) => {
      if (!dataset || dataset.kind !== 'dataset') throw new Error('불러온 데이터 블록이 필요합니다.');
      const name = n.params.className;
      const classIndex = dataset.classes.indexOf(name);
      if (classIndex < 0) throw new Error(`'${name}' 종류가 현재 로드되어 있지 않습니다.`);
      return { kind: 'datasetClass', name, classIndex, classes: [...dataset.classes] };
    }
  },

  chooseSample: {
    title: '샘플 고르기', kind: 'data', inputs: ['종류'], description: '선택한 종류에서 한 장의 샘플을 고른다.',
    formula: n => `sample[${n.params.index}]`,
    controls: [{ key: 'index', label: '샘플 번호 (0~9999)', type: 'number', step: '1', default: 0 }],
    compute: (n, [klass]) => {
      if (!klass || klass.kind !== 'datasetClass') throw new Error('종류 고르기 블록이 필요합니다.');
      const api = window.quickDrawDataset;
      const index = Math.floor(Number(n.params.index));
      const image = api.getNormalizedImage(klass.name, index);
      return { kind: 'sample', className: klass.name, classIndex: klass.classIndex, classes: [...klass.classes], index, image };
    }
  },

  datasetSampleByIndex: {
    title: '데이터 샘플', kind: 'data', inputs: ['데이터', '번호'],
    description: '불러온 모든 종류를 번갈아가며 번호에 해당하는 샘플 하나를 꺼낸다.',
    formula: () => 'sample(D, i)',
    compute: (node, [dataset, indexValue]) => {
      if (!dataset || dataset.kind !== 'dataset') throw new Error('불러온 데이터 블록이 필요합니다.');
      if (typeof indexValue !== 'number') throw new Error('샘플 번호는 숫자여야 합니다.');
      const api = window.quickDrawDataset;
      const classes = dataset.classes;
      if (!classes.length) throw new Error('데이터 종류가 없습니다.');
      const i = Math.max(0, Math.floor(indexValue));
      const classIndex = i % classes.length;
      const sampleIndex = Math.floor(i / classes.length) % 10000;
      const className = classes[classIndex];
      const image = api.getNormalizedImage(className, sampleIndex);
      return { kind: 'sample', className, classIndex, classes, index: sampleIndex, image };
    }
  },

  sampleImage: {
    title: '그림값', kind: 'data', inputs: ['샘플'], description: '샘플의 28×28 픽셀을 0~1 숫자 배열로 꺼낸다.',
    formula: () => 'x ∈ R^(28×28)',
    compute: (node, [sample]) => {
      if (handDrawingSampleOverride) return handDrawingSampleOverride();
      if (!sample || sample.kind !== 'sample') throw new Error('샘플 고르기 블록이 필요합니다.');
      return arrayValue(sample.image, [28, 28]);
    }
  },

  sampleLabel: {
    title: '종류 번호', kind: 'data', inputs: ['샘플'], description: '현재 로드된 종류 순서에서 이 샘플의 번호를 꺼낸다.',
    formula: () => 'y',
    compute: (n, [sample]) => {
      if (!sample || sample.kind !== 'sample') throw new Error('샘플 고르기 블록이 필요합니다.');
      return sample.classIndex;
    }
  },

  sampleOneHot: {
    title: '종류 벡터', kind: 'data', inputs: ['샘플'], description: '정답 종류만 1이고 나머지는 0인 벡터를 만든다.',
    formula: () => '[0,…,1,…,0]',
    compute: (node, [sample]) => {
      if (!sample || sample.kind !== 'sample') throw new Error('샘플 고르기 블록이 필요합니다.');
      const length = sample.classes.length;
      const key = `${length}:${sample.classIndex}`;
      let value = ONE_HOT_CACHE.get(key);
      if (!value) {
        const data = new Float32Array(length);
        data[sample.classIndex] = 1;
        value = arrayValue(data, [length]);
        ONE_HOT_CACHE.set(key, value);
      }
      return value;
    }
  },

  // ---------- 생성 ----------
  constantVector: {
    title: '같은 값 벡터', kind: 'generator', inputs: [], description: '같은 숫자로 채운 벡터를 만든다.',
    formula: n => `[${n.params.value}, …] (${n.params.length})`,
    controls: [{ key: 'length', label: '길이', type: 'number', step: '1', default: 3 }, { key: 'value', label: '값', type: 'number', step: 'any', default: 0 }],
    compute: node => {
      const length = Math.max(1, Math.floor(Number(node.params.length) || 1));
      const scalar = Number(node.params.value);
      return cachedNodeValue(CONSTANT_VECTOR_CACHE, node, `${length}|${scalar}`, () => {
        const data = new Float32Array(length);
        data.fill(scalar);
        return arrayValue(data, [length]);
      });
    }
  },

  randomVector: {
    title: '무작위 벡터', kind: 'generator', inputs: [], description: '-1~1 사이의 고정된 무작위 벡터를 만든다.',
    formula: n => `random(${n.params.length}, seed=${n.params.seed})`,
    controls: [{ key: 'length', label: '길이', type: 'number', step: '1', default: 3 }, { key: 'seed', label: '시드', type: 'number', step: '1', default: 1 }],
    compute: node => {
      const length = Math.max(1, Math.floor(Number(node.params.length) || 1));
      const seed = Number(node.params.seed) || 1;
      return cachedNodeValue(RANDOM_VECTOR_CACHE, node, `${length}|${seed}`, () => deterministicRandomVector(length, seed));
    }
  },

  matrix: {
    title: '행렬 만들기', kind: 'generator', inputs: [], description: '원하는 크기의 상수 또는 무작위 행렬을 만든다.',
    formula: node => `A ∈ R^(${node.params.rows}×${node.params.cols})`,
    controls: [
      { key: 'rows', label: '행 수', type: 'number', step: '1', default: 3 },
      { key: 'cols', label: '열 수', type: 'number', step: '1', default: 3 },
      { key: 'init', label: '초기값', type: 'select', default: 'random', options: [{ value: 'constant', label: '같은 값' }, { value: 'random', label: '무작위' }] },
      { key: 'value', label: '같은 값', type: 'number', step: 'any', default: 0 },
      { key: 'seed', label: '무작위 시드', type: 'number', step: '1', default: 1 },
      { key: 'scale', label: '무작위 범위', type: 'number', step: 'any', default: 0.05 }
    ],
    compute: node => {
      const rows = Math.max(1, Math.floor(Number(node.params.rows) || 1));
      const cols = Math.max(1, Math.floor(Number(node.params.cols) || 1));
      const init = String(node.params.init || 'constant');
      const value = Number(node.params.value);
      const seed = Number(node.params.seed) || 1;
      const rawScale = Number(node.params.scale);
      const scale = Number.isFinite(rawScale) ? rawScale : 0.05;
      const signature = `${rows}|${cols}|${init}|${value}|${seed}|${scale}`;
      return cachedNodeValue(MATRIX_CACHE, node, signature, () => {
        if (init === 'random') return randomArray([rows, cols], seed, scale);
        const data = new Float32Array(rows * cols);
        data.fill(value);
        return arrayValue(data, [rows, cols]);
      });
    }
  },

  // ---------- 변환 ----------
  resize28: {
    title: '28×28 변환', kind: 'transform', inputs: ['그림'], description: '그림판 입력을 28×28 값으로 바꾼다.',
    formula: () => 'T_(28×28)(image)',
    compute: (n, [input]) => {
      if (!input || input.kind !== 'image') throw new Error('그림 입력이 필요합니다.');
      const value = preprocessCanvasTo28(input.canvas);
      renderPreview(value.data);
      return value;
    }
  },

  flatten: {
    title: '펼치기', kind: 'transform', inputs: ['x'], description: '배열을 한 줄의 벡터로 펼친다.',
    formula: () => 'flatten(x)',
    compute: (node, [input]) => {
      const arr = asArrayValue(input);
      return arrayValue(arr.data, [arr.data.length]);
    },
    vjp: ([a], output, upstream) => [isArrayValue(a) ? arrayValue(new Float32Array(asArrayValue(upstream).data), a.shape) : upstream]
  },

  reshape: {
    title: '배열 모양 바꾸기', kind: 'transform', inputs: ['x'],
    description: '원소 순서는 그대로 두고 배열의 모양만 바꾼다. -1을 한 번 쓰면 나머지 크기를 자동 계산한다.',
    formula: node => `reshape(x, ${String(node.params.shape || '28,28').replace(/,/g, '×')})`,
    controls: [{ key: 'shape', label: '새 모양 (예: 26,26 또는 -1,9)', type: 'text', default: '28,28' }],
    compute: (node, [input]) => reshapeValues(input, node.params.shape),
    vjp: ([a], output, upstream) => [reshapeBackward(a, upstream)]
  },

  unfold2d: {
    title: '슬라이딩 창 펼치기', kind: 'transform', inputs: ['2차원 배열'],
    description: '2차원 배열에서 일정한 크기의 창을 움직이며 각 부분을 한 줄로 펼쳐 행렬로 만든다. 합성곱과 지역 연산을 일반 행렬 연산으로 표현할 수 있다.',
    formula: node => `unfold(x, ${node.params.kernelRows}×${node.params.kernelCols}, stride=${node.params.strideRows}×${node.params.strideCols}, pad=${node.params.padding})`,
    controls: [
      { key: 'kernelRows', label: '창 높이', type: 'number', step: '1', default: 3 },
      { key: 'kernelCols', label: '창 너비', type: 'number', step: '1', default: 3 },
      { key: 'strideRows', label: '세로 간격', type: 'number', step: '1', default: 1 },
      { key: 'strideCols', label: '가로 간격', type: 'number', step: '1', default: 1 },
      { key: 'padding', label: '바깥 0 채우기', type: 'number', step: '1', default: 0 }
    ],
    compute: (node, [input]) => unfold2dValues(node, input),
    vjp: ([a], output, upstream) => [unfold2dBackward(a, output, upstream)]
  },

  // ---------- 연산 ----------
  add: op2('더하기', 'a + b', addValues,
    ([a, b], output, upstream) => [unbroadcast(upstream, a), unbroadcast(upstream, b)]),
  subtract: op2('빼기', 'a - b', subtractValues,
    ([a, b], output, upstream) => [unbroadcast(upstream, a), unbroadcast(negateValue(upstream), b)]),
  multiply: op2('곱하기', 'a × b', multiplyValues,
    ([a, b], output, upstream) => [unbroadcast(multiplyValues(upstream, b), a), unbroadcast(multiplyValues(upstream, a), b)]),
  divide: op2('나누기', 'a ÷ b', divideValues,
    ([a, b], output, upstream) => [
      unbroadcast(divideValues(upstream, b), a),
      unbroadcast(negateValue(divideValues(multiplyValues(upstream, a), multiplyValues(b, b))), b)
    ]),
  maximum: op2('최댓값', 'max(a,b)', maximumValues,
    ([a, b], output, upstream) => {
      const fa = elementwiseBinary(a, b, (x, y) => x > y ? 1 : x < y ? 0 : 0.5);
      const fb = elementwiseBinary(a, b, (x, y) => x < y ? 1 : x > y ? 0 : 0.5);
      return [unbroadcast(multiplyValues(upstream, fa), a), unbroadcast(multiplyValues(upstream, fb), b)];
    }),
  exp: op1('지수', 'eˣ', expValues,
    ([a], output, upstream) => [multiplyValues(upstream, output)]),
  log: op1('로그', 'ln(x)', logValues,
    ([a], output, upstream) => [divideValues(upstream, a)]),

  sum: {
    title: '합', kind: 'operation', inputs: ['x'], description: '배열의 원소를 모두 더한다.', formula: () => 'Σxᵢ',
    compute: (n, [x]) => typeof x === 'number' ? x : sumArray(asArrayValue(x)),
    vjp: ([a], output, upstream) => [fillLike(a, typeof upstream === 'number' ? upstream : sumArray(asArrayValue(upstream)))]
  },

  dot: {
    title: '내적', kind: 'operation', inputs: ['a', 'b'], description: '같은 길이의 두 벡터의 내적을 구한다.',
    formula: () => 'a·b = Σaᵢbᵢ',
    compute: (n, [a, b]) => dotValues(a, b),
    vjp: ([a, b], output, upstream) => [multiplyValues(b, upstream), multiplyValues(a, upstream)]
  },

  arrayMax: {
    title: '배열 최댓값', kind: 'operation', inputs: ['배열'],
    description: '배열의 모든 원소 중 가장 큰 값 하나를 반환한다. 안정화된 지수 계산이나 Softmax 같은 수식을 직접 만들 때 사용할 수 있다.',
    formula: () => 'maxᵢ xᵢ',
    compute: (node, [value]) => arrayMaximumValue(value),
    vjp: (inputs, output, upstream) => {
      const input = asArrayValue(inputs[0]);
      const maximum = Number(output);
      let tieCount = 0;
      for (const value of input.data) if (value === maximum) tieCount++;
      if (!tieCount) return [zerosLike(input)];

      const upstreamScalar = typeof upstream === 'number' ? upstream : sumArray(asArrayValue(upstream));
      const share = upstreamScalar / tieCount;
      const grad = new Float32Array(input.data.length);
      for (let i = 0; i < input.data.length; i++) {
        if (input.data[i] === maximum) grad[i] = share;
      }
      return [arrayValue(grad, input.shape)];
    }
  },

  equal: {
    title: '같음?', kind: 'operation', inputs: ['a', 'b'],
    description: '숫자나 배열을 원소별로 비교한다. 같으면 1, 다르면 0을 반환한다.',
    formula: () => '[a = b]',
    compute: (node, [a, b]) => equalValues(a, b),
    // Equality is a discrete comparison; use zero derivative almost everywhere.
    vjp: inputs => [zerosLike(inputs[0]), zerosLike(inputs[1])]
  },

  matvec: {
    title: '행렬 × 벡터', kind: 'operation', inputs: ['행렬', '벡터'],
    description: '행렬과 벡터를 곱한다. 층 전체의 가중합도 이 연산 하나로 표현할 수 있다.',
    formula: () => 'Ax',
    compute: (node, [matrix, vector]) => matrixVectorValues(matrix, vector),
    vjp: ([matrix, vector], output, upstream) => {
      const up = asArrayValue(upstream);
      return [outerProductValues(up, asArrayValue(vector)), transposeMatrixVectorValues(matrix, up)];
    }
  },

  outerProduct: {
    title: '외적', kind: 'operation', inputs: ['a', 'b'],
    description: '두 벡터의 외적 a⊗b를 계산해 행렬을 만든다. 종류별 확률 누적 같은 계산에도 사용할 수 있다.',
    formula: () => 'a ⊗ b',
    compute: (node, [a, b]) => outerProductValues(a, b),
    vjp: ([a, b], output, upstream) => {
      const aa = asArrayValue(a);
      const bb = asArrayValue(b);
      const matrix = asArrayValue(upstream);
      if (matrix.shape.length !== 2 || matrix.shape[0] !== aa.data.length || matrix.shape[1] !== bb.data.length) {
        throw new Error('외적의 미분 행렬 크기가 맞지 않습니다.');
      }
      return [matrixVectorValues(matrix, bb), transposeMatrixVectorValues(matrix, aa)];
    }
  },

  // ---------- 실행 제어 ----------
  setVariable: {
    title: '값 바꾸기', kind: 'operation', inputs: ['새 값'],
    description: '지정한 변수의 값을 새 값으로 바꾼다. 반복 안에서는 한 회차가 끝날 때 동시에 반영된다.',
    formula: node => `${node.params.variable} ← 새 값`,
    controls: [{ key: 'variable', label: '바꿀 변수 이름', type: 'text', default: 'w' }],
    special: 'setVariable',
    compute: (node, [value]) => writeRuntimeVariable(node.params.variable, value)
  },

  sequence: {
    title: '둘 다 계산', kind: 'operation', inputs: ['먼저', '다음'],
    description: '두 입력을 모두 계산하고 두 번째 결과를 내보낸다. 여러 값 변경을 한 반복 안에 묶을 때 사용한다.',
    formula: () => '먼저 ; 다음',
    compute: (node, [first, second]) => second,
    vjp: (inputs, output, upstream) => [zerosLike(inputs[0]), upstream]
  },

  repeat: {
    title: '반복', kind: 'operation', inputs: ['실행할 식'],
    description: '연결된 계산을 여러 번 다시 실행한다. 반복 번호는 시작값부터 1씩 증가한다.',
    formula: node => {
      const start = repeatStartValue(node);
      const count = repeatCount(node);
      const end = count > 0 ? start + count - 1 : start;
      return `repeat ${count} times (${node.params.indexVariable || 'i'}=${start}…${end})`;
    },
    controls: [
      { key: 'count', label: '반복 횟수', type: 'number', step: '1', default: 10, max: MAX_REPEAT_COUNT },
      { key: 'start', label: '반복 시작값', type: 'number', step: '1', default: 0 },
      { key: 'indexVariable', label: '반복 번호 변수', type: 'text', default: 'i' }
    ],
    special: 'repeat'
  },

  derivative: {
    title: '미분', kind: 'calculus', inputs: ['식'],
    description: '연결된 식을 지정한 변수에 대해 미분한다. 숫자 출력 식의 기울기를 역방향으로 계산한다.',
    formula: n => `∂(식)/∂${n.params.variable}`,
    controls: [{ key: 'variable', label: '미분할 변수 이름', type: 'text', default: 'x' }],
    special: 'derivative'
  },

  // ---------- 확인 ----------
  display: {
    title: '값 보기', kind: 'sink', inputs: ['x'], description: '계산 결과를 그대로 보여준다.',
    formula: () => 'x',
    compute: (n, [x]) => x,
    vjp: (inputs, output, upstream) => [upstream]
  },

  // Editor-only source block. One instance stands for one external input port of
  // a collapsed 사용자 블록; it is never part of a saved definition.
  [USER_BLOCK_EXTERNAL_INPUT_TYPE]: {
    title: '외부 입력', kind: 'source', inputs: [],
    description: '사용자 블록 바깥에서 들어오는 입력입니다. 편집 모드에서만 보입니다.',
    formula: node => String(node.params.label || '입력'),
    controls: [{ key: 'label', label: '입력 이름', type: 'text', default: '입력' }],
    compute: () => 0
  }
};

function getBlockDef(type) {
  if (BLOCKS[type]) return BLOCKS[type];
  if (type.startsWith('custom:')) {
    const id = type.slice(7), custom = USER_BLOCKS.get(id);
    if (!custom) throw new Error('존재하지 않는 사용자 블록입니다.');
    return {
      title: custom.name, kind: 'custom', inputs: custom.externalInputs.map((x, i) => x.label || `입력 ${i + 1}`),
      description: `${custom.nodes.length}개 블록을 묶어 만든 사용자 블록.`, formula: () => custom.formula || custom.name,
      compute: (node, inputs) => evaluateUserDefinition(custom, inputs)
    };
  }
  throw new Error(`알 수 없는 블록: ${type}`);
}

// Optional per-block timing, installed once here beside the definitions instead
// of by whichever module happens to want the numbers. Inert until the profiler
// fills in PROFILER_HOOKS.blockCompute.
for (const [type, def] of Object.entries(BLOCKS)) {
  if (typeof def.compute !== 'function') continue;
  const compute = def.compute;
  def.compute = function(node, inputs) {
    const hook = PROFILER_HOOKS.blockCompute;
    if (!hook || !hook.active()) return compute(node, inputs);
    const started = performance.now();
    try {
      return compute(node, inputs);
    } finally {
      hook.record(type, node, performance.now() - started);
    }
  };
}
