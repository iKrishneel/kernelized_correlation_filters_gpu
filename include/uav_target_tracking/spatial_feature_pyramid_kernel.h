
#pragma once
#ifndef _SPATIAL_FEATURE_PYRAMID_KERNEL_H_
#define _SPATIAL_FEATURE_PYRAMID_KERNEL_H_

#include <uav_target_tracking/cuda_common.h>

bool spatialFeaturePyramidGPU(float **, const float *, const int *,
                              const int, const int, const int, const int,
                              const int, const int);

#endif /* _SPATIAL_FEATURE_PYRAMID_KERNEL_H_ */
