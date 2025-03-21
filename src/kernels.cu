#include <stdio.h>
#include "xcuda.h"

#ifdef __INTELLISENSE__
#define KARGS(...)
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif


//__global__ void axpy_kernel(int N, int ALPHA, int* X, int OFFX, int INCX, int* Y, int OFFY, int INCY)
//{
//    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
//    if (i < N) Y[OFFY + i * INCY] += ALPHA * X[OFFX + i * INCX];
//}

//__global__ void ker(int* a, int* x, int* y) {
//    int i = threadIdx.x;
//    y[i] = a[i] * x[i] + y[i];
//    return;
//}

__global__ void vecAdd(int* a, int* b, int* c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
    return;
}

void test_cuda(void) {
    int a[] = { 1, 2, 3 };
    int b[] = { 4, 5, 6 };
    int c[] = { 0, 0, 0 };

    int* cuda_a = 0;
    int* cuda_b = 0;
    int* cuda_c = 0;

    cudaMalloc(&cuda_a, sizeof(a));
    cudaMalloc(&cuda_b, sizeof(b));
    cudaMalloc(&cuda_c, sizeof(c));

    cudaMemcpy(cuda_a, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(b), cudaMemcpyHostToDevice);

    vecAdd KARGS(1, sizeof(a) / sizeof(int)) (cuda_a, cuda_b, cuda_c);

    cudaMemcpy(c, cuda_c, sizeof(c), cudaMemcpyDeviceToHost);
    
    printf("%d, %d, %d", c[0], c[1], c[2]);
}