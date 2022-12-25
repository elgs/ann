export const activations = {
  sigmoid: (input: number) => 1 / (1 + Math.exp(-input)),
  relu: (input: number) => Math.max(0, input),
};

export const arrayFunctions = {
  multiply: (a0: number[], a1: number[]) => a0.reduce((a, c, i) => a + c * a1[i], 0),
};