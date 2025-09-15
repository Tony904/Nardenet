#ifdef GPU

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "xcuda.h"
#include "xallocs.h"
#include "utils.h"



#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#define __shfl_down_sync(...) ( __VA_ARGS__ )
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif


__global__ void loss_mae_kernel(float* grads, float* output, float* truth, float* errors, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		float delta = output[index] - truth[index];
		errors[index] = delta;
		grads[index] = ((delta > 0.0F) ? 1.0F : 0.0F) + ((delta < 0.0F) ? -1.0F : 0.0F);  // no branching
	}
}
void launch_loss_mae_kernel(float* grads, float* output, float* truth, float* errors, int n, int batch_size) {
	n = n * batch_size;
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	loss_mae_kernel KARGS(grid_size, BLOCKSIZE) (grads, output, truth, errors, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void loss_mse_kernel(float* grads, float* output, float* truth, float* errors, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		float delta = output[index] - truth[index];
		errors[index] = delta * delta;
		grads[index] = delta;
	}
}
void launch_loss_mse_kernel(float* grads, float* output, float* truth, float* errors, int n, int batch_size) {
	n = n * batch_size;
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	loss_mse_kernel KARGS(grid_size, BLOCKSIZE) (grads, output, truth, errors, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void loss_cce_kernel(float* grads, float* output, float* truth, float* errors, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		float t = truth[index];
		float p = output[index];
		grads[index] = p - t;
		errors[index] = (t) ? -logf(p + FLT_MIN) : 0.0F;
	}
}
void launch_loss_cce_kernel(float* grads, float* output, float* truth, float* errors, int n, int batch_size) {
	n = n * batch_size;
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	loss_cce_kernel KARGS(grid_size, BLOCKSIZE) (grads, output, truth, errors, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void loss_bce_kernel(float* grads, float* output, float* truth, float* errors, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		float t = truth[index];
		float p = output[index];
		grads[index] = p - t;
		p += FLT_MIN;
		errors[index] = -t * logf(p) - (1.0F - t) * logf(1.0F - p);
	}
}
void launch_loss_bce_kernel(float* grads, float* output, float* truth, float* errors, int n, int batch_size) {
	n = n * batch_size;
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	loss_cce_kernel KARGS(grid_size, BLOCKSIZE) (grads, output, truth, errors, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void loss_l1_kernel(float* weights, int n, float decay, float* loss) {
	__shared__ float shared[BLOCKSIZE >> 5];

	int tid = threadIdx.x;
	int lane = threadIdx.x & 31;
	int warp_id = threadIdx.x >> 5;

	float val = 0.0F;
	for (int i = tid; i < n; i += BLOCKSIZE) {
		val += weights[i] < 0 ? -weights[i] : weights[i];
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
			*loss += s * decay;
		}
	}
}
void launch_loss_l1_kernel(float* weights, size_t n, float decay, float* loss) {
	loss_l1_kernel KARGS(1, BLOCKSIZE) (weights, (int)n, decay, loss);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void loss_l2_kernel(float* weights, int n, float decay, float* loss) {
	__shared__ float shared[BLOCKSIZE >> 5];

	int tid = threadIdx.x;
	int lane = threadIdx.x & 31;
	int warp_id = threadIdx.x >> 5;

	float val = 0.0F;
	for (int i = tid; i < n; i += BLOCKSIZE) {
		val += powf(weights[i], 2.0F);
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
			*loss = s * decay;
		}
	}
}
void launch_loss_l2_kernel(float* weights, size_t n, float decay, float* loss) {
	loss_l2_kernel KARGS(1, BLOCKSIZE) (weights, (int)n, decay, loss);
	CHECK_CUDA(cudaPeekAtLastError());
}

#endif