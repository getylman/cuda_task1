#include "CommonKernels.cuh"
#define SHAREDSIZE 1024

__global__ void Reduction(float *g_idata, float *g_odata, int numElements) {
  extern __shared__ float sdata[];

  const unsigned loc_id = threadIdx.x;
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[loc_id] = (index < numElements) * g_idata[index];
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (loc_id < stride) {
      sdata[loc_id] += sdata[loc_id + stride];
    }
    __syncthreads();
  }

  if (loc_id == 0) {
    atomicAdd(g_odata, sdata[0]);
  }
}

__global__ void BlockMultiplication(int numElements, float *vector1,
                                    float *vector2, float *result) {
  extern __shared__ float sdata[];

  const unsigned loc_id = threadIdx.x;
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[loc_id] = (index < numElements) * vector1[index] * vector2[index];

  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (loc_id < stride) {
      sdata[loc_id] += sdata[loc_id + stride];
    }
    __syncthreads();
  }

  if (loc_id == 0) {
    atomicAdd(result, sdata[0]);
  }
}