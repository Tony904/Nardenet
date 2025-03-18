#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void axpy_kernel(int N, float ALPHA, float* X, int OFFX, int INCX, float* Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) Y[OFFY + i * INCY] += ALPHA * X[OFFX + i * INCX];
}

__global__ void kernel(float* a, float* x, float* y) {
    int i = threadIdx.x;
    y[i] = a[i] * x[i] + y[i];
    return;
}

void test_cuda(void) {
    float a[] = { 0.1, 0.2, 0.3 };
    float b[] = { 1.1, 1.2, 1.3 };
    float c[] = { 2.1, 2.2, 2.3 };

    float* cuda_a = 0;
    float* cuda_b = 0;
    float* cuda_c = 0;

    cudaMalloc(&cuda_a, sizeof(a));
    cudaMalloc(&cuda_b, sizeof(b));
    cudaMalloc(&cuda_c, sizeof(c));

    cudaMemcpy(cuda_a, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(b), cudaMemcpyHostToDevice);

    kernel <<< 1, sizeof(a) / sizeof(float) >>> (cuda_a, cuda_b, cuda_c);

    cudaMemcpy(c, cuda_c, sizeof(c), cudaMemcpyDeviceToHost);
}