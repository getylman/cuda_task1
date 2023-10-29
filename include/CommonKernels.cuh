#pragma once

__global__ void Reduction(float *g_idata, float *g_odata, int numElements);

__global__ void BlockMultiplication(int numElements, float *vector1,
                                    float *vector2, float *result);