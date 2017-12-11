
#pragma once
#ifndef _CUDA_COMMON_H_
#define _CUDA_COMMON_H_

#include <stdio.h>
#include <math.h>

//! invoked by gslicr
// #include <curand.h>
// #include <curand_kernel.h>
#include <cufft.h>
#include <cublas_v2.h>

#define GRID_SIZE 16
#define CNN_FILTER_SIZE 256


__host__ __device__
struct caffeFilterInfo {
    int width;
    int height;
    int channels;
    int data_lenght;  //! total lenght = blob->count()
    caffeFilterInfo(int w = -1, int h = -1,
                    int c = -1, int  l = 0) :
       width(w), height(h), channels(c), data_lenght(l) {}
};


#define CUDA_ERROR_CHECK(process) {                    \
      cudaAssert((process), __FILE__, __LINE__);       \
   }                                                   \


__host__
void cuAssert(cudaError_t, char *, int, bool);

__host__ __device__ __align__(16)
int cuDivUp(int a, int b);


#endif  /* _CUDA_COMMON_H_ */

