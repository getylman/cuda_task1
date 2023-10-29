#include <fstream>
#include <iostream>
#include <stdlib.h>

__global__ void KernelMatrixAdd(int height, int width, int pitch, float *A,
                                float *B, float *result) {
  size_t index = blockDim.x * blockIdx.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;
  const size_t len = height * width;
  for (size_t i = index; i < len; i += step) {
    result[i] = A[i] + B[i];
    __syncthreads();
  }
}

void fill(float *x, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    x[i] = 1.0f;
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << "Invalid num of arguments\n";
    return 1;
  }
  float *d_x, *d_y, *d_result, *h_x, *h_y, *h_result;
  const unsigned height = atoi(argv[1]), width = atoi(argv[2]);
  const size_t len = height * width;
  const unsigned threadPerBlock = atoi(argv[3]);
  const unsigned blockPerGrid = (len + threadPerBlock - 1) / threadPerBlock;
  cudaEvent_t start, stop;
  h_x = (float *)malloc(len * sizeof(float));
  h_y = (float *)malloc(len * sizeof(float));
  h_result = (float *)malloc(len * sizeof(float));
  fill(h_x, len);
  fill(h_y, len);
  cudaMalloc(&d_x, len * sizeof(float));
  cudaMalloc(&d_y, len * sizeof(float));
  cudaMalloc(&d_result, len * sizeof(float));
  cudaMemcpy(d_x, h_x, len * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, len * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  KernelMatrixAdd<<<blockPerGrid, threadPerBlock>>>(height, width, 0, d_x, d_y,
                                                    d_result);
  cudaEventRecord(stop);

  float milliseconds = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_result, d_result, len * sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_result);
  // for (unsigned i = 0; i < height; ++i) {
  //   for (unsigned j = 0; j < width; ++j) {
  //     std::cout << h_result[i * width + j] << ' ';
  //   }
  //   std::cout << '\n';
  // }
  // std::cout << '\n';
  // std::cout << milliseconds << '\n';
  free(h_x);
  free(h_y);
  free(h_result);
  const std::string graph_data = "graph_data2.txt";
  std::ofstream file;
  file.open(graph_data, std::ios::app);
  if (file.is_open()) {
    file << (std::to_string(milliseconds) + "\n").data();
    file.close();
  }
  return 0;
}
