#include <CosineVector.cuh>
#include <ScalarMulRunner.cuh>
#include <math.h>
#include <stdio.h>

float CosineVector(int numElements, float *vector1, float *vector2,
                   int blockSize) {
  float length1 = 0.0f;
  float length2 = 0.0f;
  float result = 0.0f;

  length1 = ScalarMulSumPlusReduction(numElements, vector1, vector1, blockSize);
  length2 = ScalarMulSumPlusReduction(numElements, vector2, vector2, blockSize);
  result = ScalarMulSumPlusReduction(numElements, vector1, vector2, blockSize);
  cudaDeviceSynchronize();

  float lngtAmulB = sqrt(length1) * sqrt(length2);
  if (lngtAmulB == 0) {
    return 0.0f;
  } else {
    return result / lngtAmulB;
  }
}
