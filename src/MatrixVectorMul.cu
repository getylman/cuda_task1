#include <MatrixVectorMul.cuh>

__global__ void MatrixVectorMul(int height, int width, float *matrix,
                                float *vector, float *result) {
  size_t index = blockDim.x * blockIdx.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;
  for (unsigned i = index; i < height; i += step) {
    result[i] = 0;
    for (unsigned j = 0; j < width; ++j) {
      result[i] += matrix[i * width + j] * vector[j];
    }
  }
}
