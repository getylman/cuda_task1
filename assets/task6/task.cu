#include <fstream>
#include <iostream>
#include <stdlib.h>

#define BLOCKSIZE 256

/*
Примечание:
  Желательно самому в начале скомпилисть с разными размерами блока и нахвать все
  объектники соответствующими названиями a32.out, a64.out и тд до 1024 потому
  что размер shared памяти должен быть известен на этапе компиляции.
*/

__global__ void MatrixMul(int heightA, int widthA, int widthB, float *matrixA,
                          float *matrixB, float *matrixResult) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned loc_index = threadIdx.x;
  for (unsigned i = 0; i < heightA; ++i) {
    for (unsigned j = 0; j < widthB; ++j) {
      matrixResult[i * widthB + j] = 0;
      __shared__ float shared_arr[BLOCKSIZE];
      shared_arr[loc_index] = 0;
      if (tid < widthA) {
        shared_arr[loc_index] =
            matrixA[i * widthA + tid] * matrixB[tid * widthB + j];
      }
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
    }
  }
}

void fill(float *x, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    x[i] = 1.0f;
  }
}

void pr(const float *a, unsigned h, unsigned w) {
  for (unsigned i = 0; i < h; ++i) {
    for (unsigned j = 0; j < w; ++j) {
      std::cout << a[i * w + j] << ' ';
    }
    std::cout << '\n';
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << "Invalid num of arguments\n";
    return 1;
  }
  float *d_x, *d_y, *d_result, *h_x, *h_y, *h_result;
  const unsigned heightA = atoi(argv[1]), widthA = atoi(argv[2]),
                 widthB = atoi(argv[3]);
  const size_t len1 = heightA * widthA, len2 = widthA * widthB,
               len3 = heightA * widthB;
  const unsigned threadPerBlock = BLOCKSIZE;
  const unsigned blockPerGrid = (widthA + threadPerBlock - 1) / threadPerBlock;
  cudaEvent_t start, stop;
  h_x = (float *)malloc(len1 * sizeof(float));
  h_y = (float *)malloc(len2 * sizeof(float));
  h_result = (float *)malloc(len3 * sizeof(float));
  fill(h_x, len1);
  fill(h_y, len2);
  cudaMalloc(&d_x, len1 * sizeof(float));
  cudaMalloc(&d_y, len2 * sizeof(float));
  cudaMalloc(&d_result, len3 * sizeof(float));
  cudaMemcpy(d_x, h_x, len1 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, len2 * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  MatrixMul<<<blockPerGrid, threadPerBlock>>>(heightA, widthA, widthB, d_x, d_y,
                                              d_result);
  cudaEventRecord(stop);

  float milliseconds = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_result, d_result, len3 * sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_result);

  std::cout << "res:\n";
  pr(h_result, heightA, widthB);

  free(h_x);
  free(h_y);
  free(h_result);
  return 0;
  const std::string graph_data = "graph_data6.txt";
  std::ofstream file;
  file.open(graph_data, std::ios::app);
  if (file.is_open()) {
    file << (std::to_string(milliseconds) + "\n").data();
    file.close();
  }
  return 0;
}
