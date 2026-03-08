// Description: Generate training and test data for the neural network
// Usage: deno run --allow-write gen.ts

import * as utils from './utils.ts';

/* 
input array [
  height between 4 and 6, plus minus a random value between -1 and 1,
  weight betwen 60 and 180, plus minus a random value between -10 and 10,
]

output array[
  is_male (0 or 1),
  is_female (0 or 1),
  is_child (0 or 1),
]
*/

const inBounds = [
  [3.5, 6.5], // height between 3 and 7
  [50, 170], // weight between 40 and 180
];


// the output format should be an array of objects, with in and out properties
const gen = (samples: number) => {
  const data = [];
  // generate 100 male
  for (let i = 0; i < samples; ++i) {
    const height = 6 + Math.random() * 1 - .5;
    const weight = 160 + Math.random() * 20 - 10;
    data.push({
      in0: [height, weight],
      out: [1, 0, 0],
    })
  }

  // generate 100 female
  for (let i = 0; i < samples; ++i) {
    const height = 5 + Math.random() * 1 - .5;
    const weight = 120 + Math.random() * 20 - 10;
    data.push({
      in0: [height, weight],
      out: [0, 1, 0],
    })
  }

  // generate 100 children
  for (let i = 0; i < samples; ++i) {
    const height = 4 + Math.random() * 1 - .5;
    const weight = 60 + Math.random() * 20 - 10;
    data.push({
      in0: [height, weight],
      out: [0, 0, 1],
    })
  }
  utils.shuffle(data)
  // normalize the input data
  normalize(data);
  return data;
};

const normalize = (data: { in0: number[], in?: number[] }[]) => {
  for (let i = 0; i < inBounds.length; ++i) {
    const inBound = inBounds[i];
    const min = inBound[0];
    const max = inBound[1];
    for (const d of data) {
      d.in = d.in ?? [];
      d.in[i] = (d.in0[i] - min) / (max - min); // normalize to [0, 1]
    }
  }
  return data;
};

Deno.writeTextFileSync('train.json', JSON.stringify(gen(1000), null, 2));
Deno.writeTextFileSync('test.json', JSON.stringify(gen(200), null, 2));

