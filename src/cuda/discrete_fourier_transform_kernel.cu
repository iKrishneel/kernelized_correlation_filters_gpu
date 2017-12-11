
#include <kernelized_correlation_filters_gpu/discrete_fourier_transform_kernel.h>

__global__
void cuFloatToComplexKernel(cufftComplex *d_complex,
                      const float *dev_data, const int lenght) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < lenght) {
       d_complex[offset].x = dev_data[offset];
       d_complex[offset].y = 0.0f;
    }
}

cufftComplex* convertFloatToComplexGPU(const float *dev_data,
                                       const int FILTER_BATCH,
                                       const int FILTER_SIZE) {
    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [convertFloatToComplexGPU] FAILED\n");
    }
    int LENGHT = FILTER_BATCH * FILTER_SIZE;
    const int BYTE = LENGHT * sizeof(cufftComplex);
    cufftComplex *d_complex;
    cudaMalloc(reinterpret_cast<void**>(&d_complex), BYTE);

    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                    cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    cuFloatToComplexKernel<<<grid_size, block_size>>>(
       d_complex, dev_data, LENGHT);
    return d_complex;
}

/**
 * memory reuse for tx1
 */
// cufftComplex*
bool convertFloatToComplexGPU(cufftComplex **d_complex,
                              const float *dev_data,
                              const int FILTER_BATCH,
                              const int FILTER_SIZE) {
    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [convertFloatToComplexGPU] FAILED\n");
       return false;
    }
    int LENGHT = FILTER_BATCH * FILTER_SIZE;
    // const int BYTE = LENGHT * sizeof(cufftComplex);
    // cufftComplex *d_complex;
    // cudaMalloc(reinterpret_cast<void**>(&d_complex), BYTE);

    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                    cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    cuFloatToComplexKernel<<<grid_size, block_size>>>(
       *d_complex, dev_data, LENGHT);
    return true;
}

__global__
void copyComplexRealToFloatKernel(float *d_output,
                                  const cufftComplex *d_complex,
                                  const int lenght) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < lenght) {
       d_output[offset] = d_complex[offset].x;
    }
}

float* copyComplexRealToFloatGPU(const cufftComplex* d_complex,
                                const int FILTER_BATCH,
                                const int FILTER_SIZE) {
    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [copyComplexRealToFloatGPU] FAILED\n");
    }
    int LENGHT = FILTER_SIZE * FILTER_BATCH;
    int BYTE = LENGHT * sizeof(float);

    float *d_output;
    cudaMalloc(reinterpret_cast<void**>(&d_output), BYTE);

    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                    cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);
    
    copyComplexRealToFloatKernel<<<grid_size, block_size>>>(
       d_output, d_complex, LENGHT);

    return d_output;
}

/**
 * memeory reusing for tx1
 */

bool copyComplexRealToFloatGPU(float **d_output,
                               const cufftComplex* d_complex,
                               const int FILTER_BATCH,
                               const int FILTER_SIZE) {
    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [copyComplexRealToFloatGPU] FAILED\n");
       return false;
    }
    int LENGHT = FILTER_SIZE * FILTER_BATCH;
    // int BYTE = LENGHT * sizeof(float);
    // float *d_output;
    // cudaMalloc(reinterpret_cast<void**>(&d_output), BYTE);

    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                    cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);
    
    copyComplexRealToFloatKernel<<<grid_size, block_size>>>(
       *d_output, d_complex, LENGHT);

    return true;
}

//! normalize the data array but a given factor
__global__
void normalizeByFactorKernel(float *d_data,
                             const float factor,
                             const int lenght) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < lenght) {
       d_data[offset] /= factor;
    }
}

void normalizeByFactorGPU(float *&d_data,
                          const float factor,
                          const int FILTER_BATCH,
                          const int FILTER_SIZE) {
    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [normalizeByFactorGPU] FAILED\n");
    }

    int LENGHT = FILTER_BATCH * FILTER_SIZE;
    const int BYTE = LENGHT * sizeof(float);
    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                    cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    normalizeByFactorKernel<<<grid_size, block_size>>>(
       d_data, factor, LENGHT);
}

//! normalize the data array A but a value in the array A
__global__
void getNormalizationFactorFromIndex(float *d_value,
                                     const float *d_data,
                                     const int factor_index,
                                     const int lenght) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < lenght) {
       *d_value = d_data[factor_index];

       printf("info: %d  %3.4f\n", factor_index, *d_value);
    }
}

void normalizeByFactorInArrayGPU(float *&d_data,
                                 const int factor_index,
                                 const int FILTER_BATCH,
                                 const int FILTER_SIZE) {
    if (FILTER_BATCH == 0 || FILTER_SIZE == 0 || factor_index < 0) {
       printf("\033[31m ERROR: [normalizeByFactorInArrayGPU] FAILED\n");
    }

    int LENGHT = FILTER_BATCH * FILTER_SIZE;
    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                    cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    float *d_factor;
    cudaMalloc(reinterpret_cast<void**>(&d_factor), sizeof(float));
    getNormalizationFactorFromIndex<<<1, 1>>>(
       d_factor, d_data, factor_index, 1);
    cudaDeviceSynchronize();
    
    float factor;
    cudaMemcpy(&factor, d_factor, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    normalizeByFactorKernel<<<grid_size, block_size>>>(
       d_data, factor, LENGHT);

    cudaFree(d_factor);
}


//! fast fourier transformation
cufftComplex* cuFFTC2Cprocess(cufftComplex *in_data,
                              const cufftHandle handle,
                              const int FILTER_SIZE,
                              const int FILTER_BATCH) {
    const int OUT_BYTE = FILTER_SIZE * FILTER_BATCH * sizeof(cufftComplex);
    cufftResult cufft_status;

    cufftComplex *d_output;
    cudaMalloc(reinterpret_cast<void**>(&d_output), OUT_BYTE);
    cufft_status = cufftExecC2C(handle, in_data, d_output, CUFFT_FORWARD);
    
    if (cufft_status != cudaSuccess) {
       printf("[cuFFTC2Cprocess]: cufftExecC2C failed!");
       std::exit(-1);  //! change to shutdown
    }
    return d_output;
}

/**
 * memory resuse for tx1
 */
bool cuFFTC2Cprocess(cufftComplex **d_output, cufftComplex *in_data,
                     const cufftHandle handle, const int FILTER_SIZE,
                     const int FILTER_BATCH) {
    const int OUT_BYTE = FILTER_SIZE * FILTER_BATCH * sizeof(cufftComplex);
    cufftResult cufft_status;

    // cufftComplex *d_output;
    // cudaMalloc(reinterpret_cast<void**>(&d_output), OUT_BYTE);
    cufft_status = cufftExecC2C(handle, in_data, *d_output, CUFFT_FORWARD);
    if (cufft_status != cudaSuccess) {
       printf("[cuFFTC2Cprocess]: cufftExecC2C failed!");
       return false;
    }
    return true;
}

float *invcuFFTC2CProcess(cufftComplex *d_complex,
                          const cufftHandle handle,
                          const int FILTER_SIZE,
                          const int FILTER_BATCH, bool is_normalize) {

    if (FILTER_SIZE == 0 || FILTER_BATCH == 0) {
       printf("\033[31m ERROR: [invcuFFTC2CProcess]: INPUTS = 0 \033[0m\n");
       float empty[1];
       return empty;
    }
    cufftResult cufft_status = cufftExecC2C(handle, d_complex,
                                            d_complex, CUFFT_INVERSE);
    if (cufft_status != cudaSuccess) {
       printf("inverse cufftExecC2C failed!\n");
    }

    float *d_real = copyComplexRealToFloatGPU(d_complex,
                                              FILTER_BATCH,
                                              FILTER_SIZE);
    if (is_normalize) {
       float factor = FILTER_SIZE;
       normalizeByFactorGPU(d_real, factor, FILTER_BATCH, FILTER_SIZE);
    }
    return d_real;
}


/**
 * memory reusing for tx1
 */
bool invcuFFTC2CProcess(float **d_real,
                        cufftComplex *d_complex,
                        const cufftHandle handle,
                        const int FILTER_SIZE,
                        const int FILTER_BATCH, bool is_normalize) {
   
    if (FILTER_SIZE == 0 || FILTER_BATCH == 0) {
       printf("\033[31m ERROR: [invcuFFTC2CProcess]: INPUTS = 0 \033[0m\n");
       return false;
    }
    cufftResult cufft_status = cufftExecC2C(handle, d_complex,
                                            d_complex, CUFFT_INVERSE);
    if (cufft_status != cudaSuccess) {
       printf("inverse cufftExecC2C failed!\n");
    }

    copyComplexRealToFloatGPU(d_real, d_complex, FILTER_BATCH,
                              FILTER_SIZE);
    if (is_normalize) {
       float factor = FILTER_SIZE;
       normalizeByFactorGPU(*d_real, factor, FILTER_BATCH, FILTER_SIZE);
    }
    return true;
}
