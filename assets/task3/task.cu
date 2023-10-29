#include <fstream>
#include <iostream>
#include <stdlib.h>

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
  h_y = (float *)malloc(width * sizeof(float));
  h_result = (float *)malloc(height * sizeof(float));
  fill(h_x, len);
  fill(h_y, width);
  cudaMalloc(&d_x, len * sizeof(float));
  cudaMalloc(&d_y, width * sizeof(float));
  cudaMalloc(&d_result, height * sizeof(float));
  cudaMemcpy(d_x, h_x, len * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, width * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  MatrixVectorMul<<<blockPerGrid, threadPerBlock>>>(height, width, d_x, d_y,
                                                    d_result);
  cudaEventRecord(stop);

  float milliseconds = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_result, d_result, height * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_result);
  // for (unsigned i = 0; i < width; ++i) {
  //   // for (unsigned j = 0; j < width; ++j) {
  //   std::cout << h_result[i] << ' ';
  //   // }
  //   std::cout << '\n';
  // }
  // std::cout << '\n';
  // std::cout << milliseconds << '\n';
  free(h_x);
  free(h_y);
  free(h_result);
  const std::string graph_data = "graph_data3.txt";
  std::ofstream file;
  file.open(graph_data, std::ios::app);
  if (file.is_open()) {
    file << (std::to_string(milliseconds) + "\n").data();
    file.close();
  }
  return 0;
}
