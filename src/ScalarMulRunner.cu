#include <stdio.h>
#include <stdlib.h>

#include "CommonKernels.cuh"
#include "ScalarMul.cuh"
#include "ScalarMulRunner.cuh"

float ScalarMulTwoReductions(int numElements, float *vector1, float *vector2,
                             int blockSize) {
  float *d_x, *d_y, *d_res;
  dim3 block = dim3(blockSize);
  dim3 gridSize = dim3(
      max(1.0, ceil((float)(numElements + blockSize - 1) / (float)blockSize)));
  cudaEvent_t start, stop;
  cudaMalloc(&d_x, numElements * sizeof(float));
  cudaMalloc(&d_y, numElements * sizeof(float));
  cudaMalloc(&d_res, sizeof(float));
  cudaMemset(d_res, 0, sizeof(float));

  cudaMemcpy(d_x, vector1, numElements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, vector2, numElements * sizeof(float), cudaMemcpyHostToDevice);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  BlockMultiplication<<<gridSize, block, (blockSize * sizeof(float))>>>(
      numElements, d_x, d_y, d_res);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);

  float milliseconds = 0, res = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_res);

  return res;
}

float ScalarMulSumPlusReduction(int numElements, float *vector1, float *vector2,
                                int blockSize) {
  float *d_x, *d_y, *d_res, *d_res_sum, res_sum = 0;
  dim3 block = dim3(blockSize);
  dim3 gridSize = dim3(
      max(1.0, ceil((float)(numElements + blockSize - 1) / (float)blockSize)));
  cudaEvent_t start, stop;
  cudaMalloc(&d_x, numElements * sizeof(float));
  cudaMalloc(&d_y, numElements * sizeof(float));
  cudaMalloc(&d_res, numElements * sizeof(float));
  cudaMalloc(&d_res_sum, sizeof(float));
  cudaMemset(d_res_sum, 0, sizeof(float));

  cudaMemcpy(d_x, vector1, numElements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, vector2, numElements * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  ScalarMulBlock<<<gridSize, block>>>(numElements, d_x, d_y, d_res);

  cudaDeviceSynchronize();

  Reduction<<<gridSize, block, (blockSize * sizeof(float))>>>(d_res, d_res_sum,
                                                              numElements);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);

  float milliseconds = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(&res_sum, d_res_sum, sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_res);
  cudaFree(d_res_sum);

  return res_sum;
}