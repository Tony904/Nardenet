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
__global__ void add_arrays_kernel(float* A, float* B, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	A[index] += B[index];
}
void add_arrays_gpu(float* A, float* B, int n) {
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	add_arrays_kernel KARGS(grid_size, BLOCKSIZE) (A, B, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void copy_array_kernel(float* src, float* dst, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	dst[i] = src[i];
}
void copy_array_gpu(float* src, float* dst, int n) {
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	copy_array_kernel KARGS(grid_size, BLOCKSIZE) (src, dst, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void scale_array_kernel(float* A, float scalar, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) A[i] *= scalar;
}
void scale_array_gpu(float* A, float scalar, int n) {
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	scale_array_kernel KARGS(grid_size, BLOCKSIZE) (A, scalar, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void clamp_array_kernel(float* A, float upper, float lower, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		float val = A[i];
		if (val < 0.0F) A[i] = 0.0F;
		else if (val > 255.0F) A[i] = 255.0F;
	}
}
void clamp_array_gpu(float* A, float upper, float lower, int n) {
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	clamp_array_kernel KARGS(grid_size, BLOCKSIZE) (A, upper, lower, n);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void zero_array_kernel(float* A, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) A[i] = 0.0F;
}

void zero_array_gpu(float* A, int n) {
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	zero_array_kernel KARGS(grid_size, BLOCKSIZE) (A, n);
	CHECK_CUDA(cudaPeekAtLastError());
}