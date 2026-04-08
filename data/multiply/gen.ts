// Description: Generate training and test data for the multiplication neural network
// Usage: deno run --allow-write data/multiply/gen.ts

import * as utils from '../../utils.ts';

/*
  input array [
    a: number between 0 and 10,
    b: number between 0 and 10,
  ]

  output array (one-hot, 101 classes):
    class 0: product rounds to 0
    class 1: product rounds to 1
    ...
    class 100: product rounds to 100
*/

const inBounds = [
  [0, 10],
  [0, 10],
];

const NUM_CLASSES = 101;

const gen = (samples: number) => {
  const data = [];
  for (let i = 0; i < samples; ++i) {
    const a = Math.random() * 10;
    const b = Math.random() * 10;
    const cls = Math.min(100, Math.round(a * b));
    const out = new Array(NUM_CLASSES).fill(0);
    out[cls] = 1;
    data.push({
      in0: [a, b],
      out,
    });
  }
  utils.shuffle(data);
  normalize(data);
  return data;
};

const normalize = (data: { in0: number[]; in?: number[] }[]) => {
  for (let i = 0; i < inBounds.length; ++i) {
    const [min, max] = inBounds[i];
    for (const d of data) {
      d.in = d.in ?? [];
      d.in[i] = (d.in0[i] - min) / (max - min);
    }
  }
  return data;
};

const dir = new URL('.', import.meta.url).pathname;
Deno.writeTextFileSync(dir + 'train.json', JSON.stringify(gen(5000), null, 2));
Deno.writeTextFileSync(dir + 'test.json', JSON.stringify(gen(1000), null, 2));
console.log('Generated multiplication training and test data.');
