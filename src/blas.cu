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


// A += B
__global__ void sum_arrays_kernel(float* A, float* B, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	A[index] += B[index];
}
void sum_arrays_gpu(float* A, float* B, int n) {
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	sum_arrays_kernel KARGS(grid_size, BLOCKSIZE) (A, B, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void copy_array_kernel(float* src, float* dst, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	dst[index] = src[index];
}
void copy_array_gpu(float* src, float* dst, int n) {
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	copy_array_kernel KARGS(grid_size, BLOCKSIZE) (src, dst, n);
	CHECK_CUDA(cudaPeekAtLastError());
}