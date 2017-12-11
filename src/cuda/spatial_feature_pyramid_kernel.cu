
#include <kernelized_correlation_filters_gpu/spatial_feature_pyramid_kernel.h>


__global__ __forceinline__
void spatialFeaturePyramidKernel(float *d_output,
                                 const float *d_features,
                                 const int start_x,
                                 const int start_y,
                                 const int new_width,
                                 const int new_height,
                                 const int filter_width,
                                 const int filter_height,
                                 const int num_filters) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;
    
    if (offset < num_filters) {
       int index = 0;
       int index1 = 0;
       int index_offset = offset * filter_width * filter_height;
       int index1_offset = offset  * new_width * new_height;
       
       for (int j = start_y; j < start_y + new_height; j++) {
          for (int i = start_x; i < start_x + new_width; i++) {
             index = i + (j * filter_width) + index_offset;
             index1 = (i - start_x) + ((j - start_y) * new_width) +
                index1_offset;
             d_output[index1] = d_features[index];
          }
       }
    }
}


//! d_output = [f_w * f_h * sizeof(float), * FILTER_BATCH]
bool spatialFeaturePyramidGPU(float **d_output,
                              const float *d_features,  //! highest
                              const int *pyr_center,
                              const int new_width,
                              const int new_height,
                              const int filter_width,
                              const int filter_height,
                              const int FILTER_BATCH,
                              const int FILTER_SIZE) {
    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [convertFloatToComplexGPU] FAILED\n");
      return false;
    }
    
    const int dimension = std::ceil(std::sqrt(FILTER_BATCH));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                   cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);
    
    int start_x = pyr_center[0] - std::floor(new_width / 2);
    int start_y = pyr_center[1] - std::floor(new_height / 2);
    
    // printf("START: %d %d %d %d %d %d\n", start_x, start_y, new_width,
    //        new_height, filter_width, filter_height);
    
    spatialFeaturePyramidKernel<<<grid_size, block_size>>>(
       *d_output, d_features, start_x, start_y, new_width, new_height,
       filter_width, filter_height, FILTER_BATCH);
    
    return true;
}
