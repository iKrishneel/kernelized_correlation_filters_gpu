
#pragma once
#ifndef _SUPERPIXEL_CLUSTER_WEIGHTS_H_
#define _SUPERPIXEL_CLUSTER_WEIGHTS_H_

#include <kernelized_correlation_filters_gpu/cuda_common.h>

bool superpixelClusterWeightsGPU(float **, int**, const int *,
                                    const float *, const int, const int,
                                    const int);
bool superpixelClusterOrganizedIndicesGPU(int **, const int *, const int,
                                             const int, const int);
#endif /* _SUPERPIXEL_CLUSTER_WEIGHTS_H_ */
