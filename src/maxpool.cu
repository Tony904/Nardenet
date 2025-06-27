#include <stdio.h>
#include <math.h>
#include "xcuda.h"
#include "utils.h"


#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#define __shfl_down_sync(...) ( __VA_ARGS__ )
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif


__global__ void maxpool_kernel(float* A, float* B, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	A[index] += B[index];
}
void maxpool_gpu(float* A, float* B, int n) {
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	maxpool_kernel KARGS(grid_size, BLOCKSIZE) (A, B, n);
	CHECK_CUDA(cudaPeekAtLastError());
}