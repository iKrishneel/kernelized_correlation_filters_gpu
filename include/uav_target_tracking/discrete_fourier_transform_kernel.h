
#pragma once
#ifndef _DISCRETE_FOURIER_TRANSFORM_KERNEL_H_
#define _DISCRETE_FOURIER_TRANSFORM_KERNEL_H_

#include <uav_target_tracking/cuda_common.h>

cufftComplex* cuFFTC2Cprocess(cufftComplex *,
                              const cufftHandle,
                              const int,
                              const int);

float *invcuFFTC2CProcess(cufftComplex *d_complex,
                          const cufftHandle,
                          const int,
                          const int,
                          bool = true);

cufftComplex* convertFloatToComplexGPU(const float *,
                                       const int, const int);

float* copyComplexRealToFloatGPU(const cufftComplex*,
                                 const int, const int);

void normalizeByFactorGPU(float *&, const float,
                          const int, const int);
void normalizeByFactorInArrayGPU(float *&,
                                 const int,
                                 const int,
                                 const int);

/**
 * memory re-use for tx1
 */
bool cuFFTC2Cprocess(cufftComplex **, cufftComplex *,
                     const cufftHandle, const int,
                     const int);
bool convertFloatToComplexGPU(cufftComplex **, const float *,
                              const int, const int);
bool copyComplexRealToFloatGPU(float **, const cufftComplex*,
                               const int, const int);
bool invcuFFTC2CProcess(float **, cufftComplex *,
                        const cufftHandle, const int,
                        const int, bool = true);

#endif /* _DISCRETE_FOURIER_TRANSFORM_KERNEL_H_ */
