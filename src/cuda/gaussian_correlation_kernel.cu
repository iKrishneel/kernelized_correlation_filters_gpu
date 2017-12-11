
#include <uav_target_tracking/gaussian_correlation_kernel.h>


__device__ __forceinline__
float squaredMagnitude(const cufftComplex data) {
    return (powf(data.x, 2) + powf(data.y, 2));
}

__global__ __forceinline__
void squaredNormKernel(float *d_squared_norm,
                       const cufftComplex *d_complex,
                       const int LENGHT) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < LENGHT) {
       d_squared_norm[offset] = (d_complex[offset].x * d_complex[offset].x) +
          (d_complex[offset].y * d_complex[offset].y);
    }
}



float squaredNormGPU(const cufftComplex *d_complex,
                     const int FILTER_BATCH,
                     const int FILTER_SIZE) {
    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [squaredNormGPU] FAILED\n");
    }
    int LENGHT = FILTER_BATCH * FILTER_SIZE;
    
    float *d_squared_norm;
    const int BYTE = LENGHT * sizeof(float);
    cudaMalloc(reinterpret_cast<void**>(&d_squared_norm), BYTE);

    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                   cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);
    squaredNormKernel<<<grid_size, block_size>>>(d_squared_norm,
                                                 d_complex, LENGHT);
    
    float *d_summation;
    cudaMalloc(reinterpret_cast<void**>(&d_summation), BYTE);

    // TODO(TX1):  check and set auto
    int num_threads = 128;
    int num_blocks = 64;

    reduceSinglePass(LENGHT, num_threads, num_blocks,
                     d_squared_norm, d_summation);

    float *sum = reinterpret_cast<float*>(std::malloc(BYTE));
    cudaMemcpy(sum, d_summation, BYTE, cudaMemcpyDeviceToHost);
    
    float norm = sum[0] / FILTER_SIZE;
    
    free(sum);
    cudaFree(d_squared_norm);
    cudaFree(d_summation);

    return norm;
}

/**
 * reuable mem for tx1
 */

float squaredNormGPU(float **d_squared_norm,
                     float **d_summation,
                     const cufftComplex *d_complex,
                     const int FILTER_BATCH,
                     const int FILTER_SIZE) {
    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [squaredNormGPU] FAILED\n");
    }
    int LENGHT = FILTER_BATCH * FILTER_SIZE;
    
    // float *d_squared_norm;
    const int BYTE = LENGHT * sizeof(float);
    // cudaMalloc(reinterpret_cast<void**>(&d_squared_norm), BYTE);

    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                   cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);
    squaredNormKernel<<<grid_size, block_size>>>(*d_squared_norm,
                                                 d_complex, LENGHT);
    
    // float *d_summation;
    // cudaMalloc(reinterpret_cast<void**>(&d_summation), BYTE);

    // TODO(TX1):  check and set auto
    int num_threads = 128;
    int num_blocks = 64;

    reduceSinglePass(LENGHT, num_threads, num_blocks,
                     *d_squared_norm, *d_summation);

    float *sum = reinterpret_cast<float*>(std::malloc(BYTE));
    cudaMemcpy(sum, *d_summation, BYTE, cudaMemcpyDeviceToHost);
    
    float norm = sum[0] / FILTER_SIZE;
    
    free(sum);
    // cudaFree(d_squared_norm);
    // cudaFree(d_summation);

    return norm;
}


float* squaredNormAndMagGPU(float &norm,
                            const cufftComplex *d_complex,
                            const int FILTER_BATCH,
                            const int FILTER_SIZE) {
    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [squaredNormGPU] FAILED\n");
    }
    int LENGHT = FILTER_BATCH * FILTER_SIZE;
    
    float *d_squared_norm;  //! delete on caller side
    const int BYTE = LENGHT * sizeof(float);
    cudaMalloc(reinterpret_cast<void**>(&d_squared_norm), BYTE);

    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                   cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);
    squaredNormKernel<<<grid_size, block_size>>>(d_squared_norm,
                                                 d_complex, LENGHT);
    
    float *d_summation;
    cudaMalloc(reinterpret_cast<void**>(&d_summation), BYTE);

    // TODO(TX1):  check and set auto
    int num_threads = 128;
    int num_blocks = 64;

    reduceSinglePass(LENGHT, num_threads, num_blocks,
                     d_squared_norm, d_summation);

    float *sum = reinterpret_cast<float*>(std::malloc(BYTE));
    cudaMemcpy(sum, d_summation, BYTE, cudaMemcpyDeviceToHost);
    
    norm = sum[0] / FILTER_SIZE;
    
    free(sum);
    cudaFree(d_summation);

    return d_squared_norm;
}



/**
 * kernel for computing just the inverse
 */
__global__ __forceinline__
void complexConjuateKernel(cufftComplex *d_compl_out,
                           const cufftComplex *d_complex,
                           const int LENGHT) {

    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < LENGHT) {
       d_compl_out[offset] = d_complex[offset];
       d_compl_out[offset].y *= -1.0f;
    }
   
}

cufftComplex* complexConjuateGPU(const cufftComplex *d_complex,
                                 const int FILTER_BATCH,
                                 const int FILTER_SIZE) {

    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [invComplexConjuateGPU] FAILED\n");
    }
    
    int LENGHT = FILTER_BATCH * FILTER_SIZE;
    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                   cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    const int BYTE = LENGHT * sizeof(cufftComplex);
    cufftComplex *d_compl_out;
    cudaMalloc(reinterpret_cast<void**>(&d_compl_out), BYTE);

    complexConjuateKernel<<<grid_size, block_size>>>(d_compl_out,
                                                        d_complex, LENGHT);
    
    return d_compl_out;
}

/**
 * reusable mem for tx1
 */
bool complexConjuateGPU(cufftComplex **d_compl_out,
                        const cufftComplex *d_complex,
                        const int FILTER_BATCH,
                        const int FILTER_SIZE) {
   
    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [invComplexConjuateGPU] FAILED\n");
       return false;
    }
    
    int LENGHT = FILTER_BATCH * FILTER_SIZE;
    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                   cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    // const int BYTE = LENGHT * sizeof(cufftComplex);
    // cufftComplex *d_compl_out;
    // cudaMalloc(reinterpret_cast<void**>(&d_compl_out), BYTE);

    complexConjuateKernel<<<grid_size, block_size>>>(*d_compl_out,
                                                     d_complex, LENGHT);
    
    return true;
}


/**
 * kernel to inverse and multipy reduced into one
 */

__global__ __forceinline__
void invConjuateConvKernel(cufftComplex *d_compl_out,
                           const cufftComplex *d_complex,
                           const cufftComplex *d_compl_model,
                           const int LENGHT) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < LENGHT) {
       d_compl_out[offset].x = (
          (d_complex[offset].x * d_compl_model[offset].x) -
          (d_complex[offset].y * (d_compl_model[offset].y * -1.0f)));
       d_compl_out[offset].y = 0.0f;
    }
}

cufftComplex* invConjuateConvGPU(const cufftComplex *d_complex,
                                 const cufftComplex *d_compl_model,
                                 const int FILTER_BATCH,
                                 const int FILTER_SIZE) {

    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [invConjuateConvGPU] FAILED\n");
    }
    
    int LENGHT = FILTER_BATCH * FILTER_SIZE;
    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                   cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    const int BYTE = LENGHT * sizeof(cufftComplex);
    cufftComplex *d_compl_out;
    cudaMalloc(reinterpret_cast<void**>(&d_compl_out), BYTE);
    
    invConjuateConvKernel<<<grid_size, block_size>>>(d_compl_out, d_complex,
                                                     d_compl_model, LENGHT);
    
    return d_compl_out;
}


/**
 * inv fft over the filters
 */

__global__ __forceinline__
void invFFTSumOverFiltersKernel(float *d_xysum,
                                const float *d_real_data,
                                const int lenght,
                                const int stride,
                                const int batch) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < lenght) {
       float sum = 0.0f;
       for (int i = 0; i < batch; i++) {
          sum += (d_real_data[(i * stride) + offset]);

          // if (offset == 1) {
          //    printf(" %d %d %3.3f\n", (i * stride) + offset, stride,
          //           d_real_data[(i * stride) + offset]);
          // }
       }
       
       // d_xysum[offset] = fabsf(sum);
       d_xysum[offset] = sum;
       
       // if (offset < 5) {
       //    printf("GPU DEBUG1: %d  %3.5f %3.5f\n", offset,
       //           sum, d_xysum[offset]
       //       );
       // }
    }
}


float* invFFTSumOverFiltersGPU(const float *d_real_data,
                               const int FILTER_BATCH,
                               const int FILTER_SIZE) {

    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [invFFTSumOverFiltersGPU] FAILED\n");
       // TODO(FIX): error handling
    }
    const int OUT_BYTE = FILTER_SIZE * sizeof(float);
    
    const int dimension = std::ceil(std::sqrt(FILTER_SIZE));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                   cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);
    
    float *d_xysum;
    cudaMalloc(reinterpret_cast<void**>(&d_xysum), OUT_BYTE);

    invFFTSumOverFiltersKernel<<<grid_size, block_size>>>(
       d_xysum, d_real_data, FILTER_SIZE, FILTER_SIZE, FILTER_BATCH);
    
    return d_xysum;
}

/**
 * memory reusing for tx1
 */
bool invFFTSumOverFiltersGPU(float **d_xysum,
                               const float *d_real_data,
                               const int FILTER_BATCH,
                               const int FILTER_SIZE) {

    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [invFFTSumOverFiltersGPU] FAILED\n");
       return false;
       // TODO(FIX): error handling
    }
    // const int OUT_BYTE = FILTER_SIZE * sizeof(float);
    
    const int dimension = std::ceil(std::sqrt(FILTER_SIZE));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                   cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);
    
    // float *d_xysum;
    // cudaMalloc(reinterpret_cast<void**>(&d_xysum), OUT_BYTE);

    invFFTSumOverFiltersKernel<<<grid_size, block_size>>>(
       *d_xysum, d_real_data, FILTER_SIZE, FILTER_SIZE, FILTER_BATCH);
    
    return true;
}


/**
 * gaussian (LATER COMBINE ABOVE WITH SCALAR ADDITIONS)
 */

__global__ __forceinline__
void cuGaussianExpKernel(float *d_xysum,
                         const float xf_sqr_norm,
                         const float yf_sqr_norm,
                         const float sigma,
                         const float normalizer,
                         const int lenght) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < lenght) {
       float x = fmaxf((xf_sqr_norm + yf_sqr_norm - 2.0f *
                        d_xysum[offset]) * normalizer, 0.0f);
       d_xysum[offset] = expf(-1.0f / (sigma * sigma) * x);

       // if (offset < 10) {
       //    printf("%3.4f  %3.4f  %3.4f\n", x, d_xysum[offset],
       //           xf_sqr_norm + yf_sqr_norm - 2.0f);
       // }
    }
}

void cuGaussianExpGPU(float *&d_xysum,
                      const float xf_sqr_norm,
                      const float yf_sqr_norm,
                      const float sigma,
                      const float normalizer,
                      const int FILTER_SIZE) {
    if (normalizer == 0.0f) {
       printf("\033[31m ERROR: [cuGaussianExpGPU] FAILED: DEMO = 0\n");
       return;
    }
    
    const int dimension = std::ceil(std::sqrt(FILTER_SIZE));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                   cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    // printf("DIM: %d %3.5f %3.5f\n", FILTER_SIZE, normalizer, );

    
    cuGaussianExpKernel<<<grid_size, block_size>>>(d_xysum, xf_sqr_norm,
                                                   yf_sqr_norm, sigma,
                                                   normalizer,
                                                   FILTER_SIZE);
}
