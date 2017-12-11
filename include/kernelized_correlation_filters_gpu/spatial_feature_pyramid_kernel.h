
#pragma once
#ifndef _SPATIAL_FEATURE_PYRAMID_KERNEL_H_
#define _SPATIAL_FEATURE_PYRAMID_KERNEL_H_

#include <kernelized_correlation_filters_gpu/cuda_common.h>

bool spatialFeaturePyramidGPU(float **, const float *, const int *,
                              const int, const int, const int, const int,
                              const int, const int);

#endif /* _SPATIAL_FEATURE_PYRAMID_KERNEL_H_ */
