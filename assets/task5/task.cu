#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SHAREDSIZE 1024
#define MulTwoReduction '0'

__global__ void Reduction(float *g_idata, float *g_odata, int numElements) {
  extern __shared__ float sdata[];

  unsigned int threadi = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[threadi] = 0.0f;

  if (i < numElements) {
    sdata[threadi] = g_idata[i];
  }

  __syncthreads();

  for (unsigned int j = blockDim.x / 2; j > 0; j >>= 1) {
    if (threadi < j) {
      sdata[threadi] += sdata[threadi + j];
    }
    __syncthreads();
  }

  if (threadi == 0)
    atomicAdd(g_odata, sdata[0]);
}

__global__ void BlockMultiplication(int numElements, float *vector1,
                                    float *vector2, float *result) {
  extern __shared__ float sdata[];

  unsigned int threadi = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[threadi] = 0.0f;

  if (i < numElements) {
    sdata[threadi] = vector1[i] * vector2[i];
  }

  __syncthreads();

  for (unsigned int j = blockDim.x / 2; j > 0; j >>= 1) {
    if (threadi < j) {
      sdata[threadi] += sdata[threadi + j];
    }
    __syncthreads();
  }

  if (threadi == 0)
    atomicAdd(result, sdata[0]);
}

__global__ void ScalarMulBlock(int numElements, float *vector1, float *vector2,
                               float *result) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int gr_i = gridDim.x * blockDim.x;
  for (; i < numElements; i += gr_i) {
    result[i] = vector1[i] * vector2[i];
  }
}

float ScalarMulTwoReductions(int numElements, float *vector1, float *vector2,
                             int blockSize, float *total_time) {
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
  BlockMultiplication<<<gridSize, block, (SHAREDSIZE * sizeof(float))>>>(
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
  *total_time += milliseconds;

  return res;
}

float ScalarMulSumPlusReduction(int numElements, float *vector1, float *vector2,
                                int blockSize, float *total_time) {
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

  Reduction<<<gridSize, block, (SHAREDSIZE * sizeof(float))>>>(d_res, d_res_sum,
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
  *total_time += milliseconds;

  return res_sum;
}

float CosineVector2(int numElements, float *vector1, float *vector2,
                    int blockSize) {
  float length1 = 0.0f;
  float length2 = 0.0f;
  float result = 0.0f;
  float total_time = 0.0f;

  length1 = ScalarMulSumPlusReduction(numElements, vector1, vector1, blockSize,
                                      &total_time);
  length2 = ScalarMulSumPlusReduction(numElements, vector2, vector2, blockSize,
                                      &total_time);
  result = ScalarMulSumPlusReduction(numElements, vector1, vector2, blockSize,
                                     &total_time);
  cudaDeviceSynchronize();

  const std::string graph_data = "graph_data52.txt";
  std::ofstream file;
  file.open(graph_data, std::ios::app);
  if (file.is_open()) {
    file << (std::to_string(total_time) + "\n").data();
    file.close();
  }

  float lngtAmulB = sqrt(length1) * sqrt(length2);
  // printf("%f\n", lngtAmulB);
  if (lngtAmulB == 0) {
    return 0.0f;
  } else {
    return result / lngtAmulB;
  }
}

float CosineVector1(int numElements, float *vector1, float *vector2,
                    int blockSize) {
  float length1 = 0.0f;
  float length2 = 0.0f;
  float result = 0.0f;
  float total_time = 0.0f;

  length1 = ScalarMulTwoReductions(numElements, vector1, vector1, blockSize,
                                   &total_time);
  length2 = ScalarMulTwoReductions(numElements, vector2, vector2, blockSize,
                                   &total_time);
  result = ScalarMulTwoReductions(numElements, vector1, vector2, blockSize,
                                  &total_time);
  cudaDeviceSynchronize();

  const std::string graph_data = "graph_data51.txt";
  std::ofstream file;
  file.open(graph_data, std::ios::app);
  if (file.is_open()) {
    file << (std::to_string(total_time) + "\n").data();
    file.close();
  }

  float lngtAmulB = sqrt(length1) * sqrt(length2);
  // printf("%f\n", lngtAmulB);
  if (lngtAmulB == 0) {
    return 0.0f;
  } else {
    return result / lngtAmulB;
  }
}

void fill(float *a, unsigned len) {
  for (unsigned i = 0; i < len; ++i) {
    a[i] = 1.0f;
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("invalid number of arguments\n");
    return 1;
  }
  const size_t kLen = atoi(argv[1]);
  const size_t kBlockSize = atoi(argv[2]);
  float *A, *B;
  A = (float *)malloc(kLen * sizeof(float));
  B = (float *)malloc(kLen * sizeof(float));
  fill(A, kLen);
  fill(B, kLen);
  float res = 0;
  if (argv[3][0] == MulTwoReduction) {
    res = CosineVector1(kLen, A, B, kBlockSize);
  } else {
    res = CosineVector2(kLen, A, B, kBlockSize);
  }
  free(A);
  free(B);
}