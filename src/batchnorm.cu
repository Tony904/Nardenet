#include <stdio.h>
#include "xcuda.h"
#include "xallocs.h"
#include "utils.h"
#include <math.h>
#include "network.h"


#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#define __shfl_down_sync(...) ( __VA_ARGS__ )
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif

#define BLOCKSIZE 512



__global__ void batchnorm_kernel(float* gammas, float* betas, float* means, float* variances, float* rolling_means, float* rolling_variances, float* Z, float* Z_norm, float* act_inputs, int spatial, int batch_size, int out_n) {

	__shared__ extern float shared[];
	
	int tid = threadIdx.x;
	int filter = blockIdx.x;
	int block_size = blockDim.x;

	int fs = filter * spatial;

	// Calculate means
	shared[tid] = 0.0F;
	// Copy/Add data to shared memory
	for (int b = 0; b < batch_size; b++) {
		int offset = b * out_n + fs;
		for (int s = tid; s < spatial; s += block_size) {
			shared[tid] += Z[offset + s];
		}
	}
	__syncthreads();

	// Parallel reduction sum
	for (int stride = block_size / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			shared[tid] += shared[tid + stride];
		}
		__syncthreads();
	}

	float mean = shared[0] / (float)(spatial * batch_size);
	if (tid == 0) {
		means[filter] = mean;
		rolling_means[filter] = (mean * 0.01F) + (rolling_means[filter] * 0.99F);
	}

	// Calculate variances
	shared[tid] = 0.0F;
	for (int b = 0; b < batch_size; b++) {
		int offset = b * out_n + fs;
		for (int s = tid; s < spatial; s += block_size) {
			float dev = Z[offset + s] - mean;
			shared[tid] += dev * dev;
		}
	}
	__syncthreads();

	// Parallel reduction sum
	for (int stride = block_size / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			shared[tid] += shared[tid + stride];
		}
		__syncthreads();
	}

	float variance = shared[0] / (float)(spatial * batch_size);
	if (tid == 0) {
		variances[filter] = variance;
		rolling_variances[filter] = (variance * 0.01F) + (rolling_variances[filter] * 0.99F);
	}

	// Normalize values
	float gamma = gammas[filter];
	float beta = betas[filter];
	float sddev = sqrtf(variance + 0.00001F);
	for (int b = 0; b < batch_size; b++) {
		int offset = b * out_n + fs;
		for (int s = tid; s < spatial; s += block_size) {
			int z = offset + s;
			float znorm = (Z[z] - mean) / sddev;
			Z_norm[z] = znorm;
			act_inputs[z] = znorm * gamma + beta;
		}
	}
}

__global__ void batchnorm_kernel_warp_shuffle(float* gammas, float* betas, float* means, float* variances, float* rolling_means, float* rolling_variances, float* Z, float* Z_norm, float* act_inputs, int spatial, int batch_size, int out_n) {

	__shared__ float warp_sums[BLOCKSIZE >> 5]; // # of warps per block

	int tid = threadIdx.x;
	int filter = blockIdx.x;
	
	int lane = threadIdx.x & 31;
	int warp_id = threadIdx.x >> 5;

	int fs = filter * spatial;

	float thread_sum = 0.0F;

	// MEANS
	// Each thread computes a partial sum
	for (int b = 0; b < batch_size; b++) {
		int offset = b * out_n + fs;
		for (int s = tid; s < spatial; s += BLOCKSIZE) {
			if (s < spatial) thread_sum += Z[offset + s];
		}
	}
	// In-warp reduction using shuffle
	for (int offset = 16; offset > 0; offset >>= 1) {
		thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
	}
	// Warp leaders write to shared memory
	if (lane == 0) warp_sums[warp_id] = thread_sum;
	__syncthreads();
	// First warp reduces warp_sums[] to get block total
	float block_sum = 0.0F;
	if (warp_id == 0) {
		if (threadIdx.x < (BLOCKSIZE >> 5)) { // one thread per warp
			block_sum = warp_sums[tid];
			for (int offset = 16; offset > 0; offset >>= 1) {
				block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
			}
		}
	}
	float mean = block_sum / (float)(spatial * batch_size);
	if (tid == 0) {
		means[filter] = mean;
		rolling_means[filter] = mean * 0.01F + rolling_means[filter] * 0.99F;
	}

	// VARIANCES
	thread_sum = 0.0F;
	for (int b = 0; b < batch_size; b++) {
		int offset = b * out_n + fs;
		for (int s = tid; s < spatial; s += BLOCKSIZE) {
			float dev = Z[offset + s] - mean;
			thread_sum += dev * dev;
		}
	}
	// In-warp reduction using shuffle
	for (int offset = 16; offset > 0; offset >>= 1) {
		thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
	}
	if (lane == 0) warp_sums[warp_id] = thread_sum;
	__syncthreads();
	// First warp reduces warp_sums[] to get block total
	float block_sum = 0.0F;
	if (warp_id == 0) {
		if (tid < (BLOCKSIZE >> 5)) { // one thread per warp
			block_sum = warp_sums[tid];
			for (int offset = 16; offset > 0; offset >>= 1) {
				block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
			}
		}
	}
	float variance = block_sum / (float)(spatial * batch_size);
	if (tid == 0) {
		variances[filter] = variance;
		rolling_variances[filter] = (variance * 0.01F) + (rolling_variances[filter] * 0.99F);
	}

	// NORMALIZE AND AFFINE
	float gamma = gammas[filter];
	float beta = betas[filter];
	float sddev = sqrtf(variance + 0.00001F);
	for (int b = 0; b < batch_size; b++) {
		int offset = b * out_n + fs;
		for (int s = tid; s < spatial; s += BLOCKSIZE) {
			int z = offset + s;
			float znorm = (Z[z] - mean) / sddev;
			Z_norm[z] = znorm;
			act_inputs[z] = znorm * gamma + beta;
		}
	}
}

void test_forward_batchnorm_gpu(layer* l, size_t batch_size) {
	float* Z = l->Z;
	float* Z_norm = l->Z_norm;
	float* act_inputs = l->act_inputs;
	float* means = l->means;
	float* variances = l->variances;
	float* gammas = l->gammas;
	float* betas = l->biases;
	float* rolling_means = l->rolling_means;
	float* rolling_variances = l->rolling_variances;
	size_t F = l->n_filters;
	size_t S = l->out_w * l->out_h;
	size_t B = batch_size;
	size_t out_n = l->out_n;

	float* d_Z = 0;
	CHECK_CUDA(cudaMalloc(&d_Z, F * S * B * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(d_Z, Z, F * S * B * sizeof(float), cudaMemcpyHostToDevice));

	int grid_size = (int)F;
	int block_size = BLOCKSIZE;
	int select = 0;
	if (select == 0) {
		int shared_mem_size = (int)(S * B) * sizeof(float);
		batchnorm_kernel KARGS(grid_size, block_size, shared_mem_size) (gammas, betas, means, variances, rolling_means, rolling_variances, Z, Z_norm, act_inputs, S, batch_size, out_n);
	}
	else {
		batchnorm_kernel_warp_shuffle KARGS(grid_size, block_size) (gammas, betas, means, variances, rolling_means, rolling_variances, Z, Z_norm, act_inputs, S, batch_size, out_n);
	}
}