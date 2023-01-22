import * as utils from './utils.ts';

class Neuron {
  output: number = 0;
  delta: number = 0;
  weights: number[] = [];
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

class Net {
  inputSize: number;
  layers: Layer[];

  constructor(inputSize: number, ...layerSizes: number[]) {
    this.inputSize = inputSize;
    this.layers = [];
    for (let i = 0; i < layerSizes.length; ++i) {
      const layer = new Layer(layerSizes[i]);
      const prevLayerSize = i > 0 ? layerSizes[i - 1] : inputSize;
      for (const neuron of layer.neurons) {
        neuron.weights = new Array(prevLayerSize); // weights of each neuron in the previous layer
        for (let j = 0; j < neuron.weights.length; ++j) {
          neuron.weights[j] = Math.random() * 2 - 1;
        }
      }
      this.layers.push(layer);
    }
  }

  // update output of each neuron in a layer
  forward(prevOutputs: number[], currLayer: Layer) {
    for (const neuron of currLayer.neurons) {
      neuron.output = utils.activations.sigmoid(utils.convFunctions.multiply(prevOutputs, neuron.weights));
    }
  }

  // update output of each neuron in each layer
  forwardAll(input: number[]) {
    let prevOutputs = input;
    for (const layer of this.layers) {
      this.forward(prevOutputs, layer);
      prevOutputs = layer.neurons.map((neuron) => neuron.output);
    }
  }

  // update delta of each neuron in a layer
  backward(currLayer: Layer, nextLayer: Layer) {
    for (let i = 0; i < currLayer.neurons.length; ++i) { // for each neuron in the current layer
      const neuron = currLayer.neurons[i];
      let delta = 0;
      for (let j = 0; j < nextLayer.neurons.length; ++j) { // for each neuron in the next layer
        const nextNeuron = nextLayer.neurons[j];
        // ***MOST IMPORTANT***
        delta += nextNeuron.weights[i] * nextNeuron.delta; // sum of weights * delta of each neuron in the next layer (backpropagation)
      }
      neuron.delta = delta * neuron.output * (1 - neuron.output);
    }
  }

  // update delta of each neuron in each layer
  backwardAll(expected: number[]) {
    const lastLayer = this.layers[this.layers.length - 1];
    for (let i = 0; i < lastLayer.neurons.length; ++i) {
      const neuron = lastLayer.neurons[i];
      neuron.delta = (expected[i] - neuron.output) * neuron.output * (1 - neuron.output);
    }
    for (let i = this.layers.length - 2; i >= 0; --i) {
      this.backward(this.layers[i], this.layers[i + 1]);
    }
  }

  // update weights of each neuron in a layer
  updateWeights(currLayer: Layer, prevOutputs: number[]) {
    for (const neuron of currLayer.neurons) {
      for (let i = 0; i < neuron.weights.length; ++i) { // for each weight of each neuron in the current layer
        // ***MOST IMPORTANT***
        neuron.weights[i] += .01 * neuron.delta * prevOutputs[i]; // delta * output of each neuron in the previous layer (gradient descent)
      }
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
    this.forwardAll(input);
    this.backwardAll(expected);
    this.updateWeightsAll(input);
  }

  trainAll(data: { in: number[]; out: number[] }[]) {
    for (const [i, item] of data.entries()) {
      this.train(item.in, item.out);
    }
  }

  // predict output of each neuron in the last layer
  predict(input: number[]) {
    this.forwardAll(input);
    return this.layers[this.layers.length - 1].neurons.map((neuron) => neuron.output);
  }

  // predict output of each neuron in the last layer for each item in data
  predictAll(data: { in: number[]; out: number[] }[]) {
    const errors = data.map((item) => {
      const predicted = this.predict(item.in);
      const expected = item.out;
      const error = predicted.map((p, i) => Math.abs((p - expected[i]) / expected[i]));
      return error;
    });

    const errorsAvg = errors.reduce((acc, curr) => {
      for (let i = 0; i < acc.length; ++i) {
        acc[i] += curr[i];
      }
      return acc;
    }, new Array(errors[0].length).fill(0));
    console.log(errorsAvg.map((e) => e / errors.length));
  }

  toString() {
    return JSON.stringify(this, null, 2);
  }

  static fromString(str: string) {
    const obj = JSON.parse(str);
    const net = new Net(obj.inputSize, ...obj.layers.map((layer: Layer) => layer.neurons.length));
    net.layers = obj.layers;
    return net;
  }

}

///////////////////////////////////////////////////////////////////////////////

import train from './train.json' assert { type: 'json' };
import test from './test.json' assert { type: 'json' };
// console.log(train);
// console.log(test);

const net = new Net(4, 1);
const netStr = net.toString();
console.log(netStr);
for (let i = 0; i < 10000; ++i) {
  const epoch = i;
  console.log(`epoch: ${epoch}`);
  net.trainAll(train);
  net.predictAll(test);
}
const trainedNetStr = net.toString();

const net2 = Net.fromString(trainedNetStr);
const net2Str = net2.toString();
// console.log(net2Str);

// net.predictAll(test);
// net2.predictAll(test);
