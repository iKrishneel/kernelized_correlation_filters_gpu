
#pragma once
#ifndef _CROP_FEATURE_SPACE_H_
#define _CROP_FEATURE_SPACE_H_

#include <kernelized_correlation_filters_gpu/cuda_common.h>

void cropFeatureSpaceGPU(float **, const float *, const int,
                            const int, const int, const int, const int);

#endif /* _CROP_FEATURE_SPACE_H_ */
