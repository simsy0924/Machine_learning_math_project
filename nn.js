// 아주 작은 완전연결 신경망 코어.
// 다음 단계에서 Quick Draw 학습에 연결할 예정입니다.
class TinyNeuralNetwork {
  constructor(inputSize, hiddenSize, outputSize, learningRate = 0.01) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.learningRate = learningRate;

    this.W1 = this.randomMatrix(hiddenSize, inputSize, Math.sqrt(2 / inputSize));
    this.b1 = new Float32Array(hiddenSize);
    this.W2 = this.randomMatrix(outputSize, hiddenSize, Math.sqrt(2 / hiddenSize));
    this.b2 = new Float32Array(outputSize);
  }

  randomMatrix(rows, cols, scale) {
    return Array.from({ length: rows }, () => {
      const row = new Float32Array(cols);
      for (let j = 0; j < cols; j++) row[j] = this.randn() * scale;
      return row;
    });
  }

  randn() {
    const u = Math.max(Math.random(), 1e-12);
    const v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  relu(x) { return x > 0 ? x : 0; }

  softmax(values) {
    let max = -Infinity;
    for (const v of values) if (v > max) max = v;
    const exps = new Float32Array(values.length);
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
      exps[i] = Math.exp(values[i] - max);
      sum += exps[i];
    }
    for (let i = 0; i < exps.length; i++) exps[i] /= sum;
    return exps;
  }

  forward(input) {
    if (input.length !== this.inputSize) throw new Error("입력 크기가 맞지 않습니다.");

    const hiddenRaw = new Float32Array(this.hiddenSize);
    const hidden = new Float32Array(this.hiddenSize);
    for (let i = 0; i < this.hiddenSize; i++) {
      let s = this.b1[i];
      const row = this.W1[i];
      for (let j = 0; j < this.inputSize; j++) s += row[j] * input[j];
      hiddenRaw[i] = s;
      hidden[i] = this.relu(s);
    }

    const logits = new Float32Array(this.outputSize);
    for (let i = 0; i < this.outputSize; i++) {
      let s = this.b2[i];
      const row = this.W2[i];
      for (let j = 0; j < this.hiddenSize; j++) s += row[j] * hidden[j];
      logits[i] = s;
    }

    const probabilities = this.softmax(logits);
    return { hiddenRaw, hidden, logits, probabilities };
  }

  predict(input) {
    const { probabilities } = this.forward(input);
    let bestIndex = 0;
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > probabilities[bestIndex]) bestIndex = i;
    }
    return { classIndex: bestIndex, probabilities };
  }
}
