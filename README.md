# PCA: EXP-1  SUM ARRAY GPU
<h3>ENTER YOUR NAME</h3> <h3>YUVASREE S</h3>
<h3>ENTER YOUR REGISTER NO</h3> <h3>212225230314</h3>
<h3>EX. NO</h3> <h3>01</h3>
<h3>DATE</h3> <h3>22.10.2026</h3>
<h1> <align=center> SUM ARRAY ON HOST AND DEVICE </h3>
PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## AIM:

To perform vector addition on host and device.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1. Initialize the device and set the device properties.
2. Allocate memory on the host for input and output arrays.
3. Initialize input arrays with random values on the host.
4. Allocate memory on the device for input and output arrays, and copy input data from host to device.
5. Launch a CUDA kernel to perform vector addition on the device.
6. Copy output data from the device to the host and verify the results against the host's sequential vector addition. Free memory on the host and the device.

## PROGRAM:
```
%%writefile vector_add.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

// CUDA error checking macro
#define CHECK(call)                                                   \
{                                                                     \
    const cudaError_t error = call;                                   \
    if (error != cudaSuccess)                                         \
    {                                                                 \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);        \
        fprintf(stderr, "code: %d, reason: %s\n",                    \
                error, cudaGetErrorString(error));                    \
        exit(1);                                                      \
    }                                                                 \
}

// Timer function
double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.0e-6);
}

// Compare CPU and GPU results
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-5;
    bool match = true;

    for (int i = 0; i < N; i++)
    {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = false;
            printf("Arrays do not match!\n");
            printf("host %f gpu %f at index %d\n",
                   hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
}

// Initialize array with random values
void initialData(float *ip, int size)
{
    srand((unsigned)time(NULL));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// CPU vector addition
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

// GPU kernel
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}

int main()
{
    printf("CUDA Vector Addition Starting...\n");

    // Select GPU
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Vector size
    int nElem = 1 << 24;   // 16,777,216 elements
    printf("Vector size: %d\n", nElem);

    size_t nBytes = nElem * sizeof(float);

    // Allocate host memory
    float *h_A     = (float *)malloc(nBytes);
    float *h_B     = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    double iStart, iElaps;

    // Initialize input data
    iStart = seconds();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = seconds() - iStart;
    printf("initialData elapsed: %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // CPU computation
    iStart = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = seconds() - iStart;
    printf("CPU elapsed: %f sec\n", iElaps);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, nBytes));
    CHECK(cudaMalloc((void **)&d_B, nBytes));
    CHECK(cudaMalloc((void **)&d_C, nBytes));

    // Copy data to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threads = 512;
    dim3 block(threads);
    dim3 grid((nElem + block.x - 1) / block.x);

    // GPU computation
    iStart = seconds();
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("GPU elapsed: %f sec\n", iElaps);

    // Check for kernel errors
    CHECK(cudaGetLastError());

    // Copy result back
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // Verify results
    checkResult(hostRef, gpuRef, nElem);

    // Free device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
%%bash
nvcc -O3 vector_add.cu -o vector_add
./vector_add
```

## OUTPUT:
<img width="432" height="161" alt="image" src="https://github.com/user-attachments/assets/c22d5c16-f964-4f77-9566-b5719ade10ba" />


## RESULT:
Thus, Implementation of sum arrays on host and device is done in nvcc cuda using random number.
