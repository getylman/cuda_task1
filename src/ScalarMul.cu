#include "ScalarMul.cuh"
#define SHAREDSIZE 1024
/*
 * Calculates scalar multiplication for block
 */
__global__ void ScalarMulBlock(int numElements, float *vector1, float *vector2,
                               float *result) {
  const size_t step = gridDim.x * blockDim.x;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements;
       i += step) {
    result[i] = vector1[i] * vector2[i];
  }
}
