
#pragma once
#ifndef _GAUSSIAN_CORRELATION_KERNEL_H_
#define _GAUSSIAN_CORRELATION_KERNEL_H_

#include <kernelized_correlation_filters_gpu/cuda_common.h>
#include <kernelized_correlation_filters_gpu/threadFenceReduction_kernel.h>

float squaredNormGPU(const cufftComplex *, const int, const int);
/* returns both squared norm and mag in single call */
float* squaredNormAndMagGPU(float &, const cufftComplex *,
                            const int, const int);
/* reverse the conjuate*/
cufftComplex* complexConjuateGPU(const cufftComplex *,
                                    const int, const int);
/* sums the in complex with reverse conjuate in single call*/
cufftComplex* invConjuateConvGPU(const cufftComplex *,
                                 const cufftComplex *,
                                 const int, const int);
/* sum over filters */
float* invFFTSumOverFiltersGPU(const float *,
                               const int, const int);
/* expoential */
void cuGaussianExpGPU(float *&, const float, const float,
                      const float, const float, const int);

/* for re-usable memory in tx1 */
float squaredNormGPU(float **, float **, const cufftComplex *,
                     const int, const int);
bool complexConjuateGPU(cufftComplex **, const cufftComplex *,
                        const int, const int);
bool invFFTSumOverFiltersGPU(float **d_xysum, const float *,
                             const int, const int);

#endif /* _GAUSSIAN_CORRELATION_KERNEL_H_ */
