import * as utils from './utils.ts';

class Neuron {
  output: number = 0;
  delta: number = 0;
  activate: Function;
  weights: number[] = [];

  constructor(activate: Function = utils.activations.sigmoid) {
    this.activate = activate;
  }
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
        neuron.weights = new Array(prevLayerSize);
        for (let j = 0; j < neuron.weights.length; ++j) {
          neuron.weights[j] = Math.random() * 2 - 1;
        }
      }
      this.layers.push(layer);
    }
  }

  forward(prevOutputs: number[], currLayer: Layer) {
    for (const neuron of currLayer.neurons) {
      neuron.output = neuron.activate(utils.arrayFunctions.multiply(prevOutputs, neuron.weights));
    }
  }

  forwardAll(input: number[]) {
    let prevOutputs = input;
    for (const layer of this.layers) {
      this.forward(prevOutputs, layer);
      prevOutputs = layer.neurons.map((neuron) => neuron.output);
    }
  }

  backward(currLayer: Layer, nextLayer: Layer) {
    for (let i = 0; i < currLayer.neurons.length; ++i) {
      const neuron = currLayer.neurons[i];
      let delta = 0;
      for (let j = 0; j < nextLayer.neurons.length; ++j) {
        delta += nextLayer.neurons[j].weights[i] * nextLayer.neurons[j].delta;
      }
      neuron.delta = delta * neuron.output * (1 - neuron.output);
    }
  }

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

  updateWeights(currLayer: Layer, prevOutputs: number[]) {
    for (const neuron of currLayer.neurons) {
      for (let i = 0; i < neuron.weights.length; ++i) {
        neuron.weights[i] += 0.01 * neuron.delta * prevOutputs[i];
      }
    }
  }

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

  predict(input: number[]) {
    this.forwardAll(input);
    return this.layers[this.layers.length - 1].neurons.map((neuron) => neuron.output);
  }

  trainAll(data: { in: number[]; out: number[] }[]) {
    for (const item of data) {
      this.train(item.in, item.out);
    }
  }

  predictAll(data: { in: number[]; out: number[] }[]) {
    for (const item of data) {
      const predicted = this.predict(item.in)[0];
      const expected = item.out[0];
      const errorPercent = (expected - predicted) / expected * 100;
      const errorPercentStr = errorPercent.toFixed(2);
      console.log(`${predicted.toFixed(2)} ${expected.toFixed(2)} ${errorPercentStr}%`);
    }
  }

}

///////////////////////////////////////////////////////////////////////////////

import train from './train.json' assert { type: 'json' };
import test from './test.json' assert { type: 'json' };
// console.log(train);
// console.log(test);

const net = new Net(4, 1);

for (let i = 0; i < 10000; ++i) {
  net.trainAll(train);
}
net.predictAll(test);
