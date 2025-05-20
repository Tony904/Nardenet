#include <stdio.h>
#include "xcuda.h"
#include "xallocs.h"
#include "utils.h"
#include <math.h>
#include "network.h"


#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif


__global__ void forward_batchnorm_kernel(float* Z, int batch_size, int n_filters, int out_wh, float* mean) {
	// Steps:
	// Calculate means of each filter.
	// Calculate variances of each filter.
	// Calculate new rolling means and rolling variances.
	// Calculate z-norm then apply scale (gamma) and shift (beta).


	// calculate means
	

}

__global__ void batchnorm_kernel(float* gammas, float* betas, float* means, float* variances, float* rolling_means, float* rolling_variances, float* Z, float* Z_norm, float* act_inputs, int spatial, int batch_size, int out_n) {

	__shared__ extern float shared[];
	
	int tid = threadIdx.x;
	int filter = blockIdx.x;
	int block_size = blockDim.x;

	int fst = filter * spatial + tid;

	// Calculate means
	shared[tid] = 0.0F;
	// Copy/Add data to shared memory
	for (int b = 0; b < batch_size; b++) {
		int offset = b * out_n + fst;
		for (int s = 0; s < spatial; s += block_size) {
			if (s + tid < spatial) shared[tid] += Z[offset + s];
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
		int offset = b * out_n + fst;
		for (int s = 0; s < spatial; s += block_size) {
			if (s + tid < spatial) shared[tid] += powf(Z[offset + s] - mean, 2.0F);
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
		int offset = b * out_n + fst;
		for (int s = 0; s < spatial; s += block_size) {
			int z = offset + s;
			if (s + tid < spatial) {
				float znorm = (Z[z] - mean) / sddev;
				Z_norm[z] = znorm;
				act_inputs[z] = znorm * gamma + beta;
			}
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
	int block_size = 512;
	int shared_mem_size = (int)(S * B) * sizeof(float);
	batchnorm_kernel KARGS(grid_size, block_size, shared_mem_size) (gammas, betas, means, variances, rolling_means, rolling_variances, Z, Z_norm, act_inputs, S, batch_size, out_n);
}