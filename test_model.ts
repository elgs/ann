import { Net } from './ann.ts';

const trainedNetStr = Deno.readTextFileSync('trained.json');
const net = Net.fromString(trainedNetStr);

const labels = ['male', 'female', 'child'];

// normalize raw input using the same bounds from gen.ts
const inBounds = [
  [3.5, 6.5], // height
  [50, 170],  // weight
];

function normalize(raw: number[]): number[] {
  return raw.map((v, i) => (v - inBounds[i][0]) / (inBounds[i][1] - inBounds[i][0]));
}

// test cases: [height (feet), weight (lbs)]
const testCases = [
  { desc: 'Tall heavy (male)',     raw: [6.2, 170] },
  { desc: 'Average height (female)', raw: [5.1, 125] },
  { desc: 'Short light (child)',   raw: [3.8, 55] },
  { desc: 'Tall medium (male)',    raw: [5.8, 155] },
  { desc: 'Short medium (female)', raw: [4.8, 115] },
  { desc: 'Very short light (child)', raw: [3.6, 50] },
];

console.log('--- Testing trained model ---');
for (const tc of testCases) {
  const input = normalize(tc.raw);
  const prediction = net.predict(input);
  const predictedIndex = prediction.indexOf(Math.max(...prediction));
  const confidence = (prediction[predictedIndex] * 100).toFixed(1);
  console.log(
    `${tc.desc}: height=${tc.raw[0]}, weight=${tc.raw[1]} → ${labels[predictedIndex]} (${confidence}%) [${prediction.map((p) => p.toFixed(4)).join(', ')}]`,
  );
}
