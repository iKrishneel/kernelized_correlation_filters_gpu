
#pragma once
#ifndef _SUMMED_AREA_TABLE_H_
#define _SUMMED_AREA_TABLE_H_

#include <uav_target_tracking/cuda_common.h>

bool summedAreaTableGPU(float **,
                        const float *,
                        const int,
                        const int);
float* summedAreaTableGPU(const float *,
                          const int,
                          const int);

#endif /* _SUMMED_AREA_TABLE_H_ */
