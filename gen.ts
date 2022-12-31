// Description: Generate training and test data for the neural network
// Usage: deno run --allow-write gen.ts

import * as utils from './utils.ts';

const gen = (weights: number[]) => {
  const xs: number[] = [
    Math.random() * 2 - 1,
    Math.random() * 2 - 1,
    Math.random() * 2 - 1,
    Math.random() * 2 - 1
  ];
  const total = utils.arrayFunctions.multiply(xs, weights);
  return {
    in: xs,
    out: [
      utils.activations.sigmoid(total),
    ]
  };
}

const weights = [.4, .2, .3, .8];

const train = [];
for (let i = 0; i < 1000; ++i) {
  train.push(gen(weights));
}
Deno.writeTextFileSync('train.json', JSON.stringify(train, null, 2));

const test = [];
for (let i = 0; i < 200; ++i) {
  test.push(gen(weights));
}
Deno.writeTextFileSync('test.json', JSON.stringify(test, null, 2));
