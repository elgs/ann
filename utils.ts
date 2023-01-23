export const activations = {
  sigmoid: (input: number) => 1 / (1 + Math.exp(-input)),
  dsigmoid: function (input: number) {
    const sig = this.sigmoid(input);
    return sig * (1 - sig);
  },
  relu: (input: number) => Math.max(0, input),
  drelu: (input: number) => input > 0 ? 1 : 0,
};

export const convFunctions = {
  multiply: (a0: number[], a1: number[]) => a0.reduce((a, c, i) => a + c * a1[i], 0),

  conv2d: (
    input: number[],
    inputWidth: number,
    inputHeight: number,
    kernel: number[],
    kernelWidth: number,
    kernelHeight: number,
    strideX: number = 1,
    strideY: number = 1,
    paddingX: number = 0,
    paddingY: number = 0,
  ) => {
    if (input.length !== inputWidth * inputHeight) throw new Error('input.length !== inputWidth * inputHeight');
    if (kernel.length !== kernelWidth * kernelHeight) throw new Error('kernel.length !== kernelWidth * kernelHeight');

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
      height: outputHeight,
    };
  },

  maxPool: (
    input: number[],
    inputWidth: number,
    inputHeight: number,
    kernelWidth: number,
    kernelHeight: number,
    strideX: number = 1,
    strideY: number = 1,
    paddingX: number = 0,
    paddingY: number = 0,
  ) => {
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
      height: outputHeight,
    };
  },

  avgPool: (
    input: number[],
    inputWidth: number,
    inputHeight: number,
    kernelWidth: number,
    kernelHeight: number,
    strideX: number = 1,
    strideY: number = 1,
    paddingX: number = 0,
    paddingY: number = 0,
  ) => {
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
      height: outputHeight,
    };
  }
};

// input
// 1,2,3
// 4,5,6
// 7,8,9

// kernel
// 1,2,
// 3,4,

// manually calculated output
// 1*1+2*2+3*4+4*5 = 1+4+12+20 = 37
// 1*2+2*3+3*5+4*6 = 2+6+15+24 = 47
// 1*4+2*5+3*7+4*8 = 4+10+21+32 = 67
// 1*5+2*6+3*8+4*9 = 5+12+24+36 = 77

const printOutput = (output: number[], width: number, height: number) => {
  for (let y = 0; y < height; ++y) {
    const row = [];
    for (let x = 0; x < width; ++x) {
      row.push(output[y * width + x]);
    }
    console.log(row);
  }
  console.log();
};

const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
const kernel = [1, 2, 3, 4];

let output = convFunctions.conv2d(data, 4, 4, kernel, 2, 2, 1, 1, 1, 1);
printOutput(output.output, output.width, output.height);

output = convFunctions.maxPool(data, 4, 4, 2, 2);
printOutput(output.output, output.width, output.height);

output = convFunctions.avgPool(data, 4, 4, 2, 2);
printOutput(output.output, output.width, output.height);