#ifdef GPU

#include <stdio.h>
#include "xcuda.h"
#include <math.h>
#include "utils.h"


#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#define __shfl_down_sync(...) ( __VA_ARGS__ )
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif



__global__ void add_biases_kernel(float* arr, int spatial, float* biases) {
	__shared__ float bias;

	int channel = blockIdx.x;
	if (threadIdx.x == 0) bias = biases[channel];
	__syncthreads();

	int offset = channel * spatial;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	for (; i < spatial; i += blockDim.x) arr[offset + i] += bias;
}
void add_biases_gpu(float* arr, int spatial, float* biases, int n_filters, int batch_size) {
	add_biases_kernel KARGS(n_filters * batch_size, BLOCKSIZE) (arr, spatial, biases);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void get_bias_grads_kernel(float* bias_grads, float* grads, int spatial) {
	__shared__ float shared[BLOCKSIZE >> 5];

	int tid = threadIdx.x;
	int channel = blockIdx.x;
	int lane = threadIdx.x & 31;
	int warp_id = threadIdx.x >> 5;

	float thread_sum = 0.0F;
	int offset = channel * spatial;
	for (int s = tid; s < spatial; s += BLOCKSIZE) {
		if (s < spatial) thread_sum += grads[offset + s];
	}

	for (int offset = 16; offset > 0; offset >>= 1) {
		thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
	}

	if (lane == 0) shared[warp_id] = thread_sum;
	__syncthreads();

	float sum = 0.0F;
	if (warp_id == 0) {
		sum = (tid < (BLOCKSIZE >> 5)) ? shared[tid] : 0.0F;
		for (int offset = 16; offset > 0; offset >>= 1) {
			sum += __shfl_down_sync(0xffffffff, sum, offset);
		}
		if (tid == 0) {
			bias_grads[channel] = sum;
		}
	}
}
void get_bias_grads_gpu(float* bias_grads, float* grads, int n_filters, int spatial, int batch_size) {
	get_bias_grads_kernel KARGS(n_filters * batch_size, BLOCKSIZE) (bias_grads, grads, spatial);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void dot_product_kernel(float* A, float* B, int n, float* result) {
	__shared__ float shared[BLOCKSIZE >> 5];

	int tid = threadIdx.x;
	int lane = threadIdx.x & 31;
	int warp_id = threadIdx.x >> 5;

	float val = 0.0F;
	for (int i = tid; i < n; i += BLOCKSIZE) {
		val += A[i] * B[i];
	}

	for (int offset = 16; offset > 0; offset >>= 1) {
		val += __shfl_down_sync(0xffffffff, val, offset);
	}

	if (lane == 0) shared[warp_id] = val;
	__syncthreads();

	float s = 0.0F;
	if (warp_id == 0) {
		s = (tid < (BLOCKSIZE >> 5)) ? shared[tid] : 0.0F;
		for (int offset = 16; offset > 0; offset >>= 1) {
			s += __shfl_down_sync(0xffffffff, s, offset);
		}
		if (tid == 0) {
			*result += s;
		}
	}
}
void dot_product_gpu(float* A, float* B, int n, float* result) {
	int grid_size = GET_GRIDSIZE(1, BLOCKSIZE);
	dot_product_kernel KARGS(grid_size, BLOCKSIZE) (A, B, n, result);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void add_arrays_kernel(float* X, float* Y, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	Y[index] += X[index];
}
void add_arrays_gpu(float* X, float* Y, int n) {
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	add_arrays_kernel KARGS(grid_size, BLOCKSIZE) (X, Y, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void sum_array_kernel(float* A, int n, float* sum) {
	__shared__ float shared[BLOCKSIZE >> 5];

	int tid = threadIdx.x;
	int lane = threadIdx.x & 31;
	int warp_id = threadIdx.x >> 5;

	float val = 0.0F;
	for (int i = tid; i < n; i += BLOCKSIZE) {
		val += A[i];
	}

	for (int offset = 16; offset > 0; offset >>= 1) {
		val += __shfl_down_sync(0xffffffff, val, offset);
	}

	if (lane == 0) shared[warp_id] = val;
	__syncthreads();

	float s = 0.0F;
	if (warp_id == 0) {
		s = (tid < (BLOCKSIZE >> 5)) ? shared[tid] : 0.0F;
		for (int offset = 16; offset > 0; offset >>= 1) {
			s += __shfl_down_sync(0xffffffff, s, offset);
		}
		if (tid == 0) {
			*sum = s;
		}
	}
}
void sum_array_gpu(float* A, int n, float* sum) {
	sum_array_kernel KARGS(1, BLOCKSIZE) (A, n, sum);
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


__global__ void scale_add_array_kernel(float* src, float* dst, float* scalar, int n) {
	float s = *scalar;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) dst[i] += src[i] * s;
}
void scale_add_array_gpu(float* src, float* dst, float* scalar, int n) {
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	scale_add_array_kernel KARGS(grid_size, BLOCKSIZE) (src, dst, scalar, n);
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

#endif