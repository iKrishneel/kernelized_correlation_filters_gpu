
#include <kernelized_correlation_filters_gpu/cosine_convolution_kernel.h>

__host__ __forceinline__
void cuAssert(cudaError_t code, char *file, int line, bool abort) {
    if (code != cudaSuccess) {
       fprintf(stderr, "GPUassert: %s %s %dn",
               cudaGetErrorString(code), file, line);
       if (abort) {
          exit(code);
      }
    }
}


__host__ __device__ __align__(16)
int cuDivUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


__global__ __forceinline__
void cosineConvolutionKernel(float *d_output,
                             const float*d_cnn_codes,
                             const float *d_cos_window,
                             const int data_count) {
   
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < data_count) {
       d_output[offset] = d_cnn_codes[offset] * d_cos_window[offset];
    }
}


float* cosineConvolutionGPU(const float *d_cnn_codes,
                            const float *d_cos_window,
                            const int data_count,
                            const int BYTE) {

     const int dimension = std::ceil(std::sqrt(data_count));
     dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                     cuDivUp(dimension, GRID_SIZE));
     dim3 block_size(GRID_SIZE, GRID_SIZE);
     
     float *d_output;
     cudaMalloc(reinterpret_cast<void**>(&d_output), BYTE);
     cosineConvolutionKernel<<<grid_size, block_size>>>(
        d_output, d_cnn_codes, d_cos_window, data_count);
     
     return d_output;
}

/**
 * tx1 function
 */
bool cosineConvolutionGPU(float **d_output,
                          const float *d_cnn_codes,
                          const float *d_cos_window,
                          const int data_count) {

     const int dimension = std::ceil(std::sqrt(data_count));
     dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                     cuDivUp(dimension, GRID_SIZE));
     dim3 block_size(GRID_SIZE, GRID_SIZE);
     
     // float *d_output;
     // cudaMalloc(reinterpret_cast<void**>(&d_output), BYTE);
     cosineConvolutionKernel<<<grid_size, block_size>>>(
        *d_output, d_cnn_codes, d_cos_window, data_count);
     
     return true;
}
