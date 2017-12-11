
#pragma once
#ifndef _COSINE_CONVOLUTION_KERNEL_H_
#define _COSINE_CONVOLUTION_KERNEL_H_

#include <kernelized_correlation_filters_gpu/cuda_common.h>

float* cosineConvolutionGPU(const float *, const float *,
                          const int, const int);

bool cosineConvolutionGPU(float **,
                          const float *,
                          const float *,
                          const int);

#endif /* _COSINE_CONVOLUTION_KERNEL_H_*/

