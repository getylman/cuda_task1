#include <MatrixMul.cuh>

__global__ void MatrixMul(int heightA, int widthA, int widthB, float *matrixA,
                          float *matrixB, float *matrixResult) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned jndex = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned stepi = blockDim.x * gridDim.x;
  const unsigned stepj = blockDim.y * gridDim.y;
  float sum = 0;
  for (unsigned i = index; i < heightA; i += stepi) {
    for (unsigned j = jndex; j < widthB; j += stepj) {
      sum = 0;
      for (unsigned k = 0; k < widthA; ++k) {
        sum += matrixA[i * widthA + k] * matrixB[k * widthB + j];
      }
      matrixResult[i * widthB + j] = sum;
    }
  }
}

/*
matrixResult[i * widthB + j] = 0;
      extern __shared__ float shared_arr[];
      shared_arr[loc_index] = 0;
      shared_arr[loc_index] =
          matrixA[i * widthA + tid] * matrixB[tid * widthB + j];
      __syncthreads();
      for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (loc_index < stride) {
          shared_arr[loc_index] += shared_arr[loc_index + stride];
        }
        __syncthreads();
      }
      if (loc_index == 0) {
        atomicAdd(&matrixResult[i * widthB + j], shared_arr[0]);
      }
      __syncthreads();
*//*

for (unsigned i = 0; i < heightA; ++i) {
  for (unsigned j = 0; j < widthB; ++j) {
    matrixResult[i * widthB + j] = 0;
    for (unsigned k = 0; k < widthA; ++k) {
      matrixResult[i * widthB + j] +=
          matrixA[i * widthA + k] * matrixB[k * widthB + j];
    }
  }
}
// work good
*/