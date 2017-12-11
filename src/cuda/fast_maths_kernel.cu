
/**
 * adapted from NVIDIA cuComplex.h
 */

#include <kernelized_correlation_filters_gpu/fast_maths_kernel.h>

/**
 * multiplication
 */

__global__ __forceinline__
void multiplyComplexKernel(cufftComplex *d_results,
                           const cufftComplex *d_complex1,
                           const cufftComplex *d_complex2,
                           const int dimension) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < dimension) {
       d_results[offset].x = (d_complex1[offset].x * d_complex2[offset].x) -
          (d_complex1[offset].y * d_complex2[offset].y);
       d_results[offset].y = (d_complex1[offset].x * d_complex2[offset].y) +
          (d_complex1[offset].y * d_complex2[offset].x);
    }
}

// (a+bi)(c+di) = (a*c - b*d), (a*d+c*b)
cufftComplex* multiplyComplexGPU(const cufftComplex *d_complex1,
                                 const cufftComplex *d_complex2,
                                 const int dimension) {
    if (dimension == 0) {
       printf("ERROR: [multiplyComplexGPU] DATA DIMENSION = 0\n");
       cufftComplex empty[1];
       return empty;
    }
    
    const int csize = std::ceil(std::sqrt(dimension));
    dim3 grid_size(cuDivUp(csize, GRID_SIZE),
                   cuDivUp(csize, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    int BYTE = dimension * sizeof(cufftComplex);
    cufftComplex *d_results;
    cudaMalloc(reinterpret_cast<void**>(&d_results), BYTE);
    multiplyComplexKernel<<<grid_size, block_size>>>(d_results, d_complex1,
                                                     d_complex2, dimension);

    return d_results;
}


/**
 * memory reuse for tx1
 */
bool multiplyComplexGPU(cufftComplex **d_results,
                                 const cufftComplex *d_complex1,
                                 const cufftComplex *d_complex2,
                                 const int dimension) {
    if (dimension == 0) {
       printf("ERROR: [multiplyComplexGPU] DATA DIMENSION = 0\n");
       return false;
    }
    
    const int csize = std::ceil(std::sqrt(dimension));
    dim3 grid_size(cuDivUp(csize, GRID_SIZE),
                   cuDivUp(csize, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    int BYTE = dimension * sizeof(cufftComplex);
    // cufftComplex *d_results;
    // cudaMalloc(reinterpret_cast<void**>(&d_results), BYTE);
    multiplyComplexKernel<<<grid_size, block_size>>>(*d_results, d_complex1,
                                                     d_complex2, dimension);

    return true;
}


/**
 * mulitply by scalar
 */

__global__ __forceinline__
void multiplyComplexByScalarKernel(cufftComplex *d_results,
                                   const cufftComplex *d_complex,
                                   const cufftComplex scalar,
                                   const int dimension) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < dimension) {
       d_results[offset].x =
          (d_complex[offset].x * scalar.x) -
          (d_complex[offset].y * scalar.y);
       d_results[offset].y =
          (d_complex[offset].x * scalar.y) +
          (d_complex[offset].y * scalar.x);
    }
}

cufftComplex* multiplyComplexByScalarGPU(const cufftComplex *d_complex,
                                         const float scalar,
                                         const int dimension) {
    if (dimension == 0) {
       printf("ERROR: [multiplyComplexByScalarGPU] DATA DIMENSION = 0\n");
       cufftComplex empty[1];
       return empty;
    }
    const int csize = std::ceil(std::sqrt(dimension));
    dim3 grid_size(cuDivUp(csize, GRID_SIZE),
                   cuDivUp(csize, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    int BYTE = dimension * sizeof(cufftComplex);
    cufftComplex *d_results;
    cudaMalloc(reinterpret_cast<void**>(&d_results), BYTE);

    cufftComplex scalar_complex;
    scalar_complex.x = scalar;
    scalar_complex.y = 0.0f;

    multiplyComplexByScalarKernel<<<grid_size, block_size>>>(
       d_results, d_complex, scalar_complex, dimension);
    return d_results;
}

/**
 * memory reuse for tx1
 */
bool multiplyComplexByScalarGPU(cufftComplex **d_complex,
                                const float scalar,
                                const int dimension) {
    if (dimension == 0) {
       printf("ERROR: [multiplyComplexByScalarGPU] DATA DIMENSION = 0\n");
       return false;
    }
    const int csize = std::ceil(std::sqrt(dimension));
    dim3 grid_size(cuDivUp(csize, GRID_SIZE),
                   cuDivUp(csize, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    // int BYTE = dimension * sizeof(cufftComplex);
    // cufftComplex *d_results;
    // cudaMalloc(reinterpret_cast<void**>(&d_results), BYTE);

    cufftComplex scalar_complex;
    scalar_complex.x = scalar;
    scalar_complex.y = 0.0f;

    multiplyComplexByScalarKernel<<<grid_size, block_size>>>(
       *d_complex, *d_complex, scalar_complex, dimension);
    return true;
}



/**
 * addition kernel
 */
__global__ __forceinline__
void addComplexKernel(cufftComplex *d_results,
                      const cufftComplex *d_complex1,
                      const cufftComplex *d_complex2,
                      const int dimension) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < dimension) {
       d_results[offset].x = d_complex1[offset].x + d_complex2[offset].x;
       d_results[offset].y = d_complex1[offset].y + d_complex2[offset].y;
    }
}

cufftComplex* addComplexGPU(const cufftComplex *d_complex1,
                            const cufftComplex *d_complex2,
                            const int dimension) {
    if (dimension == 0) {
       printf("ERROR: [addComplexGPU] DATA DIMENSION = 0\n");
       cufftComplex empty[1];
       return empty;
    }
    const int csize = std::ceil(std::sqrt(dimension));
    dim3 grid_size(cuDivUp(csize, GRID_SIZE),
                   cuDivUp(csize, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    int BYTE = dimension * sizeof(cufftComplex);
    cufftComplex *d_results;
    cudaMalloc(reinterpret_cast<void**>(&d_results), BYTE);
    addComplexKernel<<<grid_size, block_size>>>(d_results, d_complex1,
                                                d_complex2, dimension);
    return d_results;
}

/**
 * memory reuse
 */

bool addComplexGPU(cufftComplex **d_complex1,
                   const cufftComplex *d_complex2,
                   const int dimension) {
    if (dimension == 0) {
       printf("ERROR: [addComplexGPU] DATA DIMENSION = 0\n");
       return false;
    }
    const int csize = std::ceil(std::sqrt(dimension));
    dim3 grid_size(cuDivUp(csize, GRID_SIZE),
                   cuDivUp(csize, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    // int BYTE = dimension * sizeof(cufftComplex);
    // cufftComplex *d_results;
    // cudaMalloc(reinterpret_cast<void**>(&d_results), BYTE);
    addComplexKernel<<<grid_size, block_size>>>(*d_complex1, *d_complex1,
                                                d_complex2, dimension);
    return true;
}


/**
 * add by scalar
 */

__global__ __forceinline__
void addComplexByScalarKernel(cufftComplex *d_results,
                              const cufftComplex *d_complex,
                              const float scalar,
                              const int dimension) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < dimension) {
       d_results[offset] = d_complex[offset];
       d_results[offset].x = d_complex[offset].x + scalar;
    }
}

cufftComplex* addComplexByScalarGPU(const cufftComplex *d_complex,
                                    const float scalar,
                                    const int dimension) {
    if (dimension == 0) {
       printf("ERROR: [addComplexByScalarGPU] DATA DIMENSION = 0\n");
       cufftComplex empty[1];
       return empty;
    }
    const int csize = std::ceil(std::sqrt(dimension));
    dim3 grid_size(cuDivUp(csize, GRID_SIZE),
                   cuDivUp(csize, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);
    
    int BYTE = dimension * sizeof(cufftComplex);
    cufftComplex *d_results;
    cudaMalloc(reinterpret_cast<void**>(&d_results), BYTE);
    addComplexByScalarKernel<<<grid_size, block_size>>>(d_results, d_complex,
                                                        scalar, dimension);
    return d_results;
}

/**
 * memory reusing tx1
 */
bool addComplexByScalarGPU(cufftComplex **d_results,
                           const cufftComplex *d_complex,
                           const float scalar,
                           const int dimension) {
    if (dimension == 0) {
       printf("ERROR: [addComplexByScalarGPU] DATA DIMENSION = 0\n");
       return false;
    }
    const int csize = std::ceil(std::sqrt(dimension));
    dim3 grid_size(cuDivUp(csize, GRID_SIZE),
                   cuDivUp(csize, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);
    
    // int BYTE = dimension * sizeof(cufftComplex);
    // cufftComplex *d_results;
    // cudaMalloc(reinterpret_cast<void**>(&d_results), BYTE);
    addComplexByScalarKernel<<<grid_size, block_size>>>(*d_results, d_complex,
                                                        scalar, dimension);
    return true;
}

/**
 * division
 */

__global__ __forceinline__
void divisionComplexKernel(cufftComplex *d_results,
                           const cufftComplex *d_complex1,
                           const cufftComplex *d_complex2,
                           const int dimension) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < dimension) {

       float s = fabsf(d_complex2[offset].x) + fabsf(d_complex2[offset].y);
       float oos = 1.0f / s;
       float ars = d_complex1[offset].x * oos;
       float ais = d_complex1[offset].y * oos;
       float brs = d_complex2[offset].x * oos;
       float bis = d_complex2[offset].y * oos;
       s = (brs * brs) + (bis * bis);
       oos = 1.0f / s;

       d_results[offset].x = ((ars * brs) + (ais * bis)) * oos;
       d_results[offset].y = ((ais * brs) - (ars * bis)) * oos;
    }
}

cufftComplex* divisionComplexGPU(const cufftComplex *d_complex1,
                                 const cufftComplex *d_complex2,
                                 const int dimension) {
    if (dimension == 0) {
       printf("ERROR: [addComplexGPU] DATA DIMENSION = 0\n");
       cufftComplex empty[1];
       return empty;
    }
    const int csize = std::ceil(std::sqrt(dimension));
    dim3 grid_size(cuDivUp(csize, GRID_SIZE),
                   cuDivUp(csize, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    int BYTE = dimension * sizeof(cufftComplex);
    cufftComplex *d_results;
    cudaMalloc(reinterpret_cast<void**>(&d_results), BYTE);
    
    divisionComplexKernel<<<grid_size, block_size>>>(d_results, d_complex1,
                                                d_complex2, dimension);
    return d_results;
}

/**
 * memory resuse on TX1
 */
bool divisionComplexGPU(cufftComplex **d_results,
                        const cufftComplex *d_complex1,
                        const cufftComplex *d_complex2,
                        const int dimension) {
    if (dimension == 0) {
       printf("ERROR: [addComplexGPU] DATA DIMENSION = 0\n");
       return false;
    }
    const int csize = std::ceil(std::sqrt(dimension));
    dim3 grid_size(cuDivUp(csize, GRID_SIZE),
                   cuDivUp(csize, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    // int BYTE = dimension * sizeof(cufftComplex);
    // cufftComplex *d_results;
    // cudaMalloc(reinterpret_cast<void**>(&d_results), BYTE);
    
    divisionComplexKernel<<<grid_size, block_size>>>(*d_results, d_complex1,
                                                     d_complex2, dimension);
    return true;
}

/**
 * absolute value
 */

__global__ __forceinline__
void absComplexKernel(cufftComplex *d_results,
                      const cufftComplex *d_complex,
                      const int dimension) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < dimension) {
       d_results[offset].x = sqrtf((d_complex[offset].x * d_complex[offset].x) +
          (d_complex[offset].y * d_complex[offset].y));
       d_results[offset].y = 0;

       // if (offset > 2 && offset < 5) {
       //    printf("DEBUG: %3.4f  %3.4f\t", d_results[offset].x,
       //    d_results[offset].y);
       //  eprintf("%3.4f  %3.4f %d\n", d_complex[offset].x,
       //    d_complex[offset].y, offset);
       // }
    }
}

bool absComplexGPU(cufftComplex **d_results,
                   const cufftComplex *d_complex,
                   const int dimension) {
    if (dimension == 0) {
       printf("ERROR: [absComplexGPU] DATA DIMENSION = 0\n");
       return false;
    }
    const int csize = std::ceil(std::sqrt(dimension));
    dim3 grid_size(cuDivUp(csize, GRID_SIZE),
                   cuDivUp(csize, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    absComplexKernel<<<grid_size, block_size>>>(*d_results,
                                                d_complex, dimension);
    
}
