
#pragma once
#ifndef _FAST_MATHS_KERNEL_H_
#define _FAST_MATHS_KERNEL_H_

#include <uav_target_tracking/cuda_common.h>


cufftComplex* multiplyComplexGPU(const cufftComplex *,
                                 const cufftComplex *,
                                 const int);

cufftComplex* multiplyComplexByScalarGPU(const cufftComplex *,
                                         const float,
                                         const int);

cufftComplex* addComplexGPU(const cufftComplex *,
                            const cufftComplex *,
                            const int);

cufftComplex* divisionComplexGPU(const cufftComplex *,
                                 const cufftComplex *,
                                 const int);

cufftComplex* addComplexByScalarGPU(const cufftComplex *,
                                    const float,
                                    const int);

/* for reusable memory on tx1 */
bool multiplyComplexGPU(cufftComplex **,
                        const cufftComplex *,
                        const cufftComplex *,
                        const int);
bool addComplexGPU(cufftComplex **,
                   const cufftComplex *,
                   const int);
bool multiplyComplexByScalarGPU(cufftComplex **,
                                const float,
                                const int);
bool addComplexByScalarGPU(cufftComplex **,
                           const cufftComplex *,
                           const float,
                           const int);
bool divisionComplexGPU(cufftComplex **,
                        const cufftComplex *,
                        const cufftComplex *,
                        const int);
bool absComplexGPU(cufftComplex **,
                   const cufftComplex *,
                   const int);

#endif /* _FAST_MATHS_KERNEL_H_ */
