#include <KernelMatrixAdd.cuh>

__global__ void KernelMatrixAdd(int height, int width, int pitch, float *A,
                                float *B, float *result) {
  size_t index = blockDim.x * blockIdx.x + threadIdx.x;
  size_t jndex = blockDim.y * blockIdx.y + threadIdx.y;
  if (jndex < width && index < height) {
    result[index * pitch + jndex] =
        A[index * pitch + jndex] + B[index * pitch + jndex];
  }
}
