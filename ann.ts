// the implementation of a simple feedforward neural network with backpropagation in TypeScript, without using any libraries, only using basic math functions and array operations

// https://www.youtube.com/watch?v=sIX_9n-1UbM

import * as utils from './utils.ts';

export { utils };
export const { activations, shuffle } = utils;

let activation = 'sigmoid'; // or 'relu'

export function setActivation(act: string) { activation = act; }
export function getActivation() { return activation; }

class Neuron {
  output: number = 0;
  delta: number = 0;
  weights: number[] = [];
  bias: number = 0;
}

class Layer {
  neurons: Neuron[];

  constructor(size: number) {
    this.neurons = [];
    for (let i = 0; i < size; ++i) {
      this.neurons.push(new Neuron());
    }
  }
}

export class Net {
  inputSize: number;
  layers: Layer[];
  learningRate: number;

  constructor(inputSize: number, layerSizes: number[], learningRate: number = 0.01) {
    this.inputSize = inputSize;
    this.layers = [];
    this.learningRate = learningRate;
    for (let i = 0; i < layerSizes.length; ++i) {
      const layer = new Layer(layerSizes[i]);
      const prevLayerSize = i > 0 ? layerSizes[i - 1] : inputSize;
      for (const neuron of layer.neurons) {
        neuron.weights = new Array(prevLayerSize); // weights of each neuron in the previous layer
        for (let j = 0; j < neuron.weights.length; ++j) {
          neuron.weights[j] = Math.random() * 2 - 1;
        }
        neuron.bias = Math.random() * 2 - 1;
      }
      this.layers.push(layer);
    }
  }

  // update output of each neuron in a layer
  forward(prevOutputs: number[], currLayer: Layer, isOutputLayer: boolean) {
    for (const neuron of currLayer.neurons) {
      const weightedSum = utils.convFunctions.multiply(prevOutputs, neuron.weights) + neuron.bias;
      if (isOutputLayer) {
        neuron.output = weightedSum;
      } else if (activation === 'sigmoid') {
        neuron.output = utils.activations.sigmoid(weightedSum);
      } else if (activation === 'relu') {
        neuron.output = utils.activations.relu(weightedSum);
      }
    }
  }

  // update output of each neuron in each layer, return the output of the last layer which is softmax
  forwardAll(input: number[]) {
    let prevOutputs = input;
    for (let i = 0; i < this.layers.length; ++i) {
      const layer = this.layers[i];
      this.forward(prevOutputs, layer, i === this.layers.length - 1);
      prevOutputs = layer.neurons.map((neuron) => neuron.output);
    }
    // at this point, prevOutputs contains the output of the last layer
    // update the output of the last layer with softmax
    const lastLayer = this.layers[this.layers.length - 1];
    const outputs = lastLayer.neurons.map((neuron) => neuron.output);
    const softmaxOutputs = utils.activations.softmax(outputs);
    return softmaxOutputs;
  }

  // update delta of each neuron in a layer
  backward(currLayer: Layer, nextLayer: Layer) {
    for (let i = 0; i < currLayer.neurons.length; ++i) { // for each neuron in the current layer
      const neuron = currLayer.neurons[i];
      let dCurrentOutput = 0
      for (let j = 0; j < nextLayer.neurons.length; ++j) { // for each neuron in the next layer
        const nextLayerNeuron = nextLayer.neurons[j];
        // ***MOST IMPORTANT***
        dCurrentOutput += nextLayerNeuron.weights[i] * nextLayerNeuron.delta; // sum of weights * delta of each neuron in the next layer (backpropagation)
      }
      if (activation === 'sigmoid') {
        neuron.delta = dCurrentOutput * utils.activations.dsigmoid(neuron.output);
      } else if (activation === 'relu') {
        neuron.delta = dCurrentOutput * utils.activations.drelu(neuron.output);
      }
    }
  }

  // update delta of each neuron in each layer
  backwardAll(softmaxOutputs: number[], expected: number[]) {
    if (expected.length !== softmaxOutputs.length) {
      throw new Error('Expected output size does not match the softmax output size, expected: ' + expected.length + ', got: ' + softmaxOutputs.length);
    }

    const dLastLayerOutputs = utils.activations.dcrossEntropySoftmax(softmaxOutputs, expected);
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
  updateWeights(currLayer: Layer, prevOutputs: number[]) {
    for (const neuron of currLayer.neurons) { // for each neuron in the current layer
      for (let i = 0; i < neuron.weights.length; ++i) { // for each weight of each neuron in the current layer
        // ***MOST IMPORTANT***
        neuron.weights[i] -= this.learningRate * neuron.delta * prevOutputs[i]; // dweight = learningRate * delta * output of the previous layer
      }
      neuron.bias -= this.learningRate * neuron.delta;
    }
  }

  // update weights of each neuron in each layer
  updateWeightsAll(input: number[]) {
    let prevOutputs = input;
    for (const layer of this.layers) {
      this.updateWeights(layer, prevOutputs);
      prevOutputs = layer.neurons.map((neuron) => neuron.output);
    }
  }

  train(input: number[], expected: number[]) {
    const softmaxOutputs = this.forwardAll(input);
    const error = utils.activations.crossEntropy(softmaxOutputs, expected);
    console.log('error:', error, 'input:', input, 'predicted:', softmaxOutputs, 'expected:', expected);
    this.backwardAll(softmaxOutputs, expected);
    this.updateWeightsAll(input);
    return error;
  }

  trainAll(data: { in: number[]; out: number[] }[]) {
    utils.shuffle(data);
    let totalError = 0;
    for (const item of data) {
      totalError += this.train(item.in, item.out);
    }
    return totalError / data.length;
  }

  // predict output of each neuron in the last layer
  predict(input: number[]) {
    return this.forwardAll(input);
  }

  argmax(values: number[]) {
    let maxIndex = 0;
    for (let i = 1; i < values.length; ++i) {
      if (values[i] > values[maxIndex]) {
        maxIndex = i;
      }
    }
    return maxIndex;
  }

  // predict output of each neuron in the last layer for each item in data
  predictAll(data: { in: number[]; out: number[] }[]) {
    const errors = data.map((item) => {
      const predicted = this.predict(item.in);
      const expected = item.out;
      const error = predicted.map((p, i) => Math.abs(p - expected[i]));
      console.log(`error: ${error.map((e) => e.toFixed(6)).join(', ')} input: ${item.in.map((v) => v.toFixed(6)).join(', ')} predicted: ${predicted.map((p) => p.toFixed(6)).join(', ')} expected: ${expected.join(', ')}`);
      return {
        error,
        loss: utils.activations.crossEntropy(predicted, expected),
        correct: this.argmax(predicted) === this.argmax(expected),
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
      accuracy,
    };
  }

  evaluate(data: { in: number[]; out: number[] }[]) {
    let totalLoss = 0, correct = 0;
    for (const d of data) {
      const pred = this.predict(d.in);
      totalLoss += utils.activations.crossEntropy(pred, d.out);
      if (this.argmax(pred) === this.argmax(d.out)) correct++;
    }
    return { avgLoss: totalLoss / data.length, accuracy: correct / data.length };
  }

  toJSON() {
    return JSON.parse(JSON.stringify({ inputSize: this.inputSize, layers: this.layers, learningRate: this.learningRate }));
  }

  static fromJSON(obj: any) {
    const net = new Net(obj.inputSize, obj.layers.map((layer: Layer) => layer.neurons.length), obj.learningRate ?? 0.01);
    net.layers = obj.layers;
    return net;
  }

  toString() {
    return JSON.stringify(this, null, 2);
  }

  static fromString(str: string) {
    const obj = JSON.parse(str);
    const net = new Net(obj.inputSize, obj.layers.map((layer: Layer) => layer.neurons.length), obj.learningRate ?? 0.01);
    net.layers = obj.layers;
    return net;
  }

}

///////////////////////////////////////////////////////////////////////////////

if (import.meta.main) {
  const { default: train } = await import('./train.json', { with: { type: 'json' } });
  const { default: test } = await import('./test.json', { with: { type: 'json' } });

  const net = new Net(2, [4, 3], 0.05);
  const netStr = net.toString();
  for (let i = 0; i < 100; ++i) {
    const epoch = i;
    const trainLoss = net.trainAll(train);
    const metrics = net.predictAll(test);
    console.log(
      `epoch ${epoch}: trainLoss=${trainLoss.toFixed(6)} testLoss=${metrics.avgLoss.toFixed(6)} accuracy=${(metrics.accuracy * 100).toFixed(2)}% avgErrors=${metrics.avgErrors.map((e) => e.toFixed(6)).join(', ')}`,
    );
  }
  const trainedNetStr = net.toString();
  Deno.writeTextFileSync('trained.json', trainedNetStr);

  const net2 = Net.fromString(trainedNetStr);
  const net2Str = net2.toString();

  console.log('netStr:', netStr);
  // console.log('net2Str:', net2Str);
  console.log('trainedNetStr:', trainedNetStr);

  console.log(net.predict([.9, .9]));
  console.log(net2.predict([.9, .9]));
}