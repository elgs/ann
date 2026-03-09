var __defProp = Object.defineProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};

// utils.ts
var utils_exports = {};
__export(utils_exports, {
  activations: () => activations,
  convFunctions: () => convFunctions,
  shuffle: () => shuffle
});
var activations = {
  sigmoid: (input) => 1 / (1 + Math.exp(-input)),
  dsigmoid: (output) => output * (1 - output),
  relu: (input) => Math.max(0, input),
  drelu: (output) => output > 0 ? 1 : 0,
  softmax: function(input) {
    const max = Math.max(...input);
    const exp = input.map((x) => Math.exp(x - max));
    const sum = exp.reduce((a, c) => a + c, 0);
    return exp.map((x) => x / sum);
  },
  crossEntropy: function(input, expected) {
    return -expected.reduce((a, c, i) => a + c * Math.log(input[i] + 1e-15), 0);
  },
  dcrossEntropySoftmax: function(softmaxOutputs, expected) {
    return softmaxOutputs.map((x, i) => x - expected[i]);
  }
};
var shuffle = (array) => {
  let currentIndex = array.length;
  while (currentIndex != 0) {
    let randomIndex = Math.floor(Math.random() * currentIndex);
    --currentIndex;
    [array[currentIndex], array[randomIndex]] = [array[randomIndex], array[currentIndex]];
  }
};
var convFunctions = {
  multiply: (a0, a1) => a0.reduce((a, c, i) => a + c * a1[i], 0),
  conv2d: (input, inputWidth, inputHeight, kernel, kernelWidth, kernelHeight, strideX = 1, strideY = 1, paddingX = 0, paddingY = 0) => {
    if (input.length !== inputWidth * inputHeight) throw new Error("input.length !== inputWidth * inputHeight");
    if (kernel.length !== kernelWidth * kernelHeight) throw new Error("kernel.length !== kernelWidth * kernelHeight");
    const outputWidth = Math.floor((inputWidth + paddingX * 2 - kernelWidth) / strideX) + 1;
    const outputHeight = Math.floor((inputHeight + paddingY * 2 - kernelHeight) / strideY) + 1;
    const output = new Array(outputWidth * outputHeight);
    for (let oy = 0; oy < outputHeight; ++oy) {
      for (let ox = 0; ox < outputWidth; ++ox) {
        let sum = 0;
        for (let ky = 0; ky < kernelHeight; ++ky) {
          for (let kx = 0; kx < kernelWidth; ++kx) {
            const ix = ox * strideX + kx - paddingX;
            const iy = oy * strideY + ky - paddingY;
            if (ix >= 0 && iy >= 0 && ix < inputWidth && iy < inputHeight) {
              sum += input[iy * inputWidth + ix] * kernel[ky * kernelWidth + kx];
            }
          }
        }
        output[oy * outputWidth + ox] = sum;
      }
    }
    return {
      output,
      width: outputWidth,
      height: outputHeight
    };
  },
  maxPool: (input, inputWidth, inputHeight, kernelWidth, kernelHeight, strideX = 1, strideY = 1, paddingX = 0, paddingY = 0) => {
    const outputWidth = Math.floor((inputWidth + paddingX * 2 - kernelWidth) / strideX) + 1;
    const outputHeight = Math.floor((inputHeight + paddingY * 2 - kernelHeight) / strideY) + 1;
    const output = new Array(outputWidth * outputHeight);
    for (let oy = 0; oy < outputHeight; ++oy) {
      for (let ox = 0; ox < outputWidth; ++ox) {
        let max = -Infinity;
        for (let ky = 0; ky < kernelHeight; ++ky) {
          for (let kx = 0; kx < kernelWidth; ++kx) {
            const ix = ox * strideX + kx - paddingX;
            const iy = oy * strideY + ky - paddingY;
            if (ix >= 0 && iy >= 0 && ix < inputWidth && iy < inputHeight) {
              max = Math.max(max, input[iy * inputWidth + ix]);
            }
          }
        }
        output[oy * outputWidth + ox] = max;
      }
    }
    return {
      output,
      width: outputWidth,
      height: outputHeight
    };
  },
  avgPool: (input, inputWidth, inputHeight, kernelWidth, kernelHeight, strideX = 1, strideY = 1, paddingX = 0, paddingY = 0) => {
    const outputWidth = Math.floor((inputWidth + paddingX * 2 - kernelWidth) / strideX) + 1;
    const outputHeight = Math.floor((inputHeight + paddingY * 2 - kernelHeight) / strideY) + 1;
    const output = new Array(outputWidth * outputHeight);
    for (let oy = 0; oy < outputHeight; ++oy) {
      for (let ox = 0; ox < outputWidth; ++ox) {
        let sum = 0;
        for (let ky = 0; ky < kernelHeight; ++ky) {
          for (let kx = 0; kx < kernelWidth; ++kx) {
            const ix = ox * strideX + kx - paddingX;
            const iy = oy * strideY + ky - paddingY;
            if (ix >= 0 && iy >= 0 && ix < inputWidth && iy < inputHeight) {
              sum += input[iy * inputWidth + ix];
            }
          }
        }
        output[oy * outputWidth + ox] = sum / (kernelWidth * kernelHeight);
      }
    }
    return {
      output,
      width: outputWidth,
      height: outputHeight
    };
  }
};

// ann.ts
var { activations: activations2, shuffle: shuffle2 } = utils_exports;
var activation = "sigmoid";
function setActivation(act) {
  activation = act;
}
function getActivation() {
  return activation;
}
var Neuron = class {
  output = 0;
  delta = 0;
  weights = [];
  bias = 0;
};
var Layer = class {
  neurons;
  constructor(size) {
    this.neurons = [];
    for (let i = 0; i < size; ++i) {
      this.neurons.push(new Neuron());
    }
  }
};
var Net = class _Net {
  inputSize;
  layers;
  learningRate;
  constructor(inputSize, layerSizes, learningRate = 0.01) {
    this.inputSize = inputSize;
    this.layers = [];
    this.learningRate = learningRate;
    for (let i = 0; i < layerSizes.length; ++i) {
      const layer = new Layer(layerSizes[i]);
      const prevLayerSize = i > 0 ? layerSizes[i - 1] : inputSize;
      for (const neuron of layer.neurons) {
        neuron.weights = new Array(prevLayerSize);
        for (let j = 0; j < neuron.weights.length; ++j) {
          neuron.weights[j] = Math.random() * 2 - 1;
        }
        neuron.bias = Math.random() * 2 - 1;
      }
      this.layers.push(layer);
    }
  }
  // update output of each neuron in a layer
  forward(prevOutputs, currLayer, isOutputLayer) {
    for (const neuron of currLayer.neurons) {
      const weightedSum = convFunctions.multiply(prevOutputs, neuron.weights) + neuron.bias;
      if (isOutputLayer) {
        neuron.output = weightedSum;
      } else if (activation === "sigmoid") {
        neuron.output = activations.sigmoid(weightedSum);
      } else if (activation === "relu") {
        neuron.output = activations.relu(weightedSum);
      }
    }
  }
  // update output of each neuron in each layer, return the output of the last layer which is softmax
  forwardAll(input) {
    let prevOutputs = input;
    for (let i = 0; i < this.layers.length; ++i) {
      const layer = this.layers[i];
      this.forward(prevOutputs, layer, i === this.layers.length - 1);
      prevOutputs = layer.neurons.map((neuron) => neuron.output);
    }
    const lastLayer = this.layers[this.layers.length - 1];
    const outputs = lastLayer.neurons.map((neuron) => neuron.output);
    const softmaxOutputs = activations.softmax(outputs);
    return softmaxOutputs;
  }
  // update delta of each neuron in a layer
  backward(currLayer, nextLayer) {
    for (let i = 0; i < currLayer.neurons.length; ++i) {
      const neuron = currLayer.neurons[i];
      let dCurrentOutput = 0;
      for (let j = 0; j < nextLayer.neurons.length; ++j) {
        const nextLayerNeuron = nextLayer.neurons[j];
        dCurrentOutput += nextLayerNeuron.weights[i] * nextLayerNeuron.delta;
      }
      if (activation === "sigmoid") {
        neuron.delta = dCurrentOutput * activations.dsigmoid(neuron.output);
      } else if (activation === "relu") {
        neuron.delta = dCurrentOutput * activations.drelu(neuron.output);
      }
    }
  }
  // update delta of each neuron in each layer
  backwardAll(softmaxOutputs, expected) {
    if (expected.length !== softmaxOutputs.length) {
      throw new Error("Expected output size does not match the softmax output size, expected: " + expected.length + ", got: " + softmaxOutputs.length);
    }
    const dLastLayerOutputs = activations.dcrossEntropySoftmax(softmaxOutputs, expected);
    const lastLayer = this.layers[this.layers.length - 1];
    for (let i = 0; i < lastLayer.neurons.length; ++i) {
      const neuron = lastLayer.neurons[i];
      neuron.delta = dLastLayerOutputs[i];
    }
    for (let i = this.layers.length - 2; i >= 0; --i) {
      this.backward(this.layers[i], this.layers[i + 1]);
    }
  }
  // update weights of each neuron in a layer
  updateWeights(currLayer, prevOutputs) {
    for (const neuron of currLayer.neurons) {
      for (let i = 0; i < neuron.weights.length; ++i) {
        neuron.weights[i] -= this.learningRate * neuron.delta * prevOutputs[i];
      }
      neuron.bias -= this.learningRate * neuron.delta;
    }
  }
  // update weights of each neuron in each layer
  updateWeightsAll(input) {
    let prevOutputs = input;
    for (const layer of this.layers) {
      this.updateWeights(layer, prevOutputs);
      prevOutputs = layer.neurons.map((neuron) => neuron.output);
    }
  }
  train(input, expected) {
    const softmaxOutputs = this.forwardAll(input);
    const error = activations.crossEntropy(softmaxOutputs, expected);
    console.log("error:", error, "input:", input, "predicted:", softmaxOutputs, "expected:", expected);
    this.backwardAll(softmaxOutputs, expected);
    this.updateWeightsAll(input);
    return error;
  }
  trainAll(data) {
    shuffle(data);
    let totalError = 0;
    for (const item of data) {
      totalError += this.train(item.in, item.out);
    }
    return totalError / data.length;
  }
  // predict output of each neuron in the last layer
  predict(input) {
    return this.forwardAll(input);
  }
  argmax(values) {
    let maxIndex = 0;
    for (let i = 1; i < values.length; ++i) {
      if (values[i] > values[maxIndex]) {
        maxIndex = i;
      }
    }
    return maxIndex;
  }
  // predict output of each neuron in the last layer for each item in data
  predictAll(data) {
    const errors = data.map((item) => {
      const predicted = this.predict(item.in);
      const expected = item.out;
      const error = predicted.map((p, i) => Math.abs(p - expected[i]));
      console.log(`error: ${error.map((e) => e.toFixed(6)).join(", ")} input: ${item.in.map((v) => v.toFixed(6)).join(", ")} predicted: ${predicted.map((p) => p.toFixed(6)).join(", ")} expected: ${expected.join(", ")}`);
      return {
        error,
        loss: activations.crossEntropy(predicted, expected),
        correct: this.argmax(predicted) === this.argmax(expected)
      };
    });
    const errorsAvg = errors.reduce((acc, curr) => {
      for (let i = 0; i < acc.length; ++i) {
        acc[i] += curr.error[i];
      }
      return acc;
    }, new Array(errors[0].error.length).fill(0));
    const avgLoss = errors.reduce((acc, curr) => acc + curr.loss, 0) / errors.length;
    const accuracy = errors.reduce((acc, curr) => acc + (curr.correct ? 1 : 0), 0) / errors.length;
    return {
      avgErrors: errorsAvg.map((e) => e / errors.length),
      avgLoss,
      accuracy
    };
  }
  evaluate(data) {
    let totalLoss = 0, correct = 0;
    for (const d of data) {
      const pred = this.predict(d.in);
      totalLoss += activations.crossEntropy(pred, d.out);
      if (this.argmax(pred) === this.argmax(d.out)) correct++;
    }
    return { avgLoss: totalLoss / data.length, accuracy: correct / data.length };
  }
  toJSON() {
    return JSON.parse(JSON.stringify({ inputSize: this.inputSize, layers: this.layers, learningRate: this.learningRate }));
  }
  static fromJSON(obj) {
    const net = new _Net(obj.inputSize, obj.layers.map((layer) => layer.neurons.length), obj.learningRate ?? 0.01);
    net.layers = obj.layers;
    return net;
  }
  toString() {
    return JSON.stringify(this, null, 2);
  }
  static fromString(str) {
    const obj = JSON.parse(str);
    const net = new _Net(obj.inputSize, obj.layers.map((layer) => layer.neurons.length), obj.learningRate ?? 0.01);
    net.layers = obj.layers;
    return net;
  }
};
if (import.meta.main) {
  const { default: train } = await import("./train.json", { with: { type: "json" } });
  const { default: test } = await import("./test.json", { with: { type: "json" } });
  const net = new Net(2, [4, 3], 0.05);
  const netStr = net.toString();
  for (let i = 0; i < 100; ++i) {
    const epoch = i;
    const trainLoss = net.trainAll(train);
    const metrics = net.predictAll(test);
    console.log(
      `epoch ${epoch}: trainLoss=${trainLoss.toFixed(6)} testLoss=${metrics.avgLoss.toFixed(6)} accuracy=${(metrics.accuracy * 100).toFixed(2)}% avgErrors=${metrics.avgErrors.map((e) => e.toFixed(6)).join(", ")}`
    );
  }
  const trainedNetStr = net.toString();
  Deno.writeTextFileSync("trained.json", trainedNetStr);
  const net2 = Net.fromString(trainedNetStr);
  const net2Str = net2.toString();
  console.log("netStr:", netStr);
  console.log("trainedNetStr:", trainedNetStr);
  console.log(net.predict([0.9, 0.9]));
  console.log(net2.predict([0.9, 0.9]));
}
export {
  Net,
  activations2 as activations,
  getActivation,
  setActivation,
  shuffle2 as shuffle
};
