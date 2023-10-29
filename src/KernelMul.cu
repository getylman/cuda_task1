#include <KernelMul.cuh>

__global__ void KernelMul(int numElements, float *x, float *y, float *result) {
  size_t index = blockDim.x * blockIdx.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;
  for (size_t i = index; i < numElements; i += step) {
    result[i] = x[i] * y[i];
  }
}
