#include <stdio.h>
#include "xcuda.h"
#include "xallocs.h"
#include "utils.h"
#include <math.h>
#include "network.h"
#include "batchnorm.h"


#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#define __shfl_down_sync(...) ( __VA_ARGS__ )
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif


__global__ void forward_batchnorm_kernel_no_shuffle(
	const float* __restrict__ gammas, const float* __restrict__ betas,
	float* means, float* variances,
	float* rolling_means, float* rolling_variances,
	float* __restrict__ Z, float* Z_norm, float* act_inputs,
	int spatial, int n, int batch_size) {

	__shared__ float shared[BLOCKSIZE];
	
	int tid = threadIdx.x;
	int filter = blockIdx.x;
	int block_size = blockDim.x;

	int fs = filter * spatial;

	// Calculate means
	shared[tid] = 0.0F;
	// Copy/Add data to shared memory
	for (int b = 0; b < batch_size; b++) {
		int offset = b * n + fs;
		for (int s = tid; s < spatial; s += block_size) {
			shared[tid] += Z[offset + s];
		}
	}
	__syncthreads();

	// Parallel reduction sum
	for (int stride = block_size >> 1; stride > 0; stride >>= 1) {
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
		int offset = b * n + fs;
		for (int s = tid; s < spatial; s += block_size) {
			float dev = Z[offset + s] - mean;
			shared[tid] += dev * dev;
		}
	}
	__syncthreads();

	// Parallel reduction sum
	for (int stride = block_size >> 1; stride > 0; stride >>= 1) {
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
		int offset = b * n + fs;
		for (int s = tid; s < spatial; s += block_size) {
			int z = offset + s;
			float znorm = (Z[z] - mean) / sddev;
			Z_norm[z] = znorm;
			act_inputs[z] = znorm * gamma + beta;
		}
	}
}

__global__ void forward_batchnorm_kernel(
	const float* __restrict__ gammas, const float* __restrict__ betas,
	float* means, float* variances,
	float* rolling_means, float* rolling_variances,
	float* __restrict__ Z, float* Z_norm, float* act_inputs,
	int spatial, int n, int batch_size) {

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
		int offset = b * n + fs;
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
		block_sum = (tid < (BLOCKSIZE >> 5)) ? warp_sums[tid] : 0.0F;
		for (int offset = 16; offset > 0; offset >>= 1) {
			block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
		}
		if (tid == 0) {
			float mean = block_sum / (float)(spatial * batch_size);
			warp_sums[0] = mean;
			means[filter] = mean;
			rolling_means[filter] = mean * 0.01F + rolling_means[filter] * 0.99F;
		}
	}
	__syncthreads();
	float mean = warp_sums[0];

	// VARIANCES
	thread_sum = 0.0F;
	for (int b = 0; b < batch_size; b++) {
		int offset = b * n + fs;
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
	block_sum = 0.0F;
	if (warp_id == 0) {
		block_sum = (tid < (BLOCKSIZE >> 5)) ? warp_sums[tid] : 0.0f;
		for (int offset = 16; offset > 0; offset >>= 1) {
			block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
		}
		if (tid == 0) {
			float variance = block_sum / (float)(spatial * batch_size);
			warp_sums[0] = variance;
			variances[filter] = variance;
			rolling_variances[filter] = (variance * 0.01F) + (rolling_variances[filter] * 0.99F);
		}
	}
	__syncthreads();
	float variance = warp_sums[0];

	// NORMALIZE AND AFFINE
	float gamma = gammas[filter];
	float beta = betas[filter];
	float sddev = sqrtf(variance + 0.00001F);
	for (int b = 0; b < batch_size; b++) {
		int offset = b * n + fs;
		for (int s = tid; s < spatial; s += BLOCKSIZE) {
			int z = offset + s;
			float znorm = (Z[z] - mean) / sddev;
			Z_norm[z] = znorm;
			act_inputs[z] = znorm * gamma + beta;
		}
	}
}

void forward_batchnorm_gpu(float* gammas, float* betas,
	float* means, float* variances,
	float* rolling_means, float* rolling_variances,
	float* Z, float* Z_norm, float* act_inputs,
	int spatial, int n_filters, int batch_size) {

	int n = spatial * n_filters;
	forward_batchnorm_kernel KARGS(n_filters, BLOCKSIZE) (gammas, betas, means, variances, rolling_means, rolling_variances, Z, Z_norm, act_inputs, spatial, batch_size, n);
	CHECK_CUDA(cudaPeekAtLastError());
}

void test_forward_batchnorm_gpu(void) {
	int batch_size = 8;
	int w = 320;
	int h = 320;
	int spatial = w * h;
	int n_filters = 64;
	int out_n = spatial * n_filters;
	float* Z = (float*)xcalloc(out_n * batch_size, sizeof(float));
	float* Z_norm = (float*)xcalloc(out_n * batch_size, sizeof(float));
	float* act_inputs = (float*)xcalloc(out_n * batch_size, sizeof(float));
	float* means = (float*)xcalloc(n_filters, sizeof(float));
	float* variances = (float*)xcalloc(n_filters, sizeof(float));
	float* gammas = (float*)xcalloc(n_filters, sizeof(float));
	float* betas = (float*)xcalloc(n_filters, sizeof(float));
	float* rolling_means = (float*)xcalloc(n_filters, sizeof(float));
	float* rolling_variances = (float*)xcalloc(n_filters, sizeof(float));

	fill_array_rand_float(Z, out_n * batch_size, 0.0F, 1.0F);
	float* d_Z = 0;
	CHECK_CUDA(cudaMalloc(&d_Z, out_n * batch_size * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(d_Z, Z, out_n * batch_size * sizeof(float), cudaMemcpyHostToDevice));

	float* d_act_inputs = 0;
	CHECK_CUDA(cudaMalloc(&d_act_inputs, out_n * batch_size * sizeof(float)));

	float* d_Z_norm = 0;
	CHECK_CUDA(cudaMalloc(&d_Z_norm, out_n * batch_size * sizeof(float)));

	fill_array_rand_float(means, n_filters, 0.0F, 1.0F);
	float* d_means = 0;
	CHECK_CUDA(cudaMalloc(&d_means, n_filters * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(d_means, means, n_filters * sizeof(float), cudaMemcpyHostToDevice));

	fill_array_rand_float(means, n_filters, 0.0F, 1.0F);
	float* d_variances = 0;
	CHECK_CUDA(cudaMalloc(&d_variances, n_filters * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(d_variances, variances, n_filters * sizeof(float), cudaMemcpyHostToDevice));

	fill_array_rand_float(gammas, n_filters, 0.0F, 1.0F);
	float* d_gammas = 0;
	CHECK_CUDA(cudaMalloc(&d_gammas, n_filters * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(d_gammas, gammas, n_filters * sizeof(float), cudaMemcpyHostToDevice));

	fill_array_rand_float(betas, n_filters, 0.0F, 1.0F);
	float* d_betas = 0;
	CHECK_CUDA(cudaMalloc(&d_betas, n_filters * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(d_betas, betas, n_filters * sizeof(float), cudaMemcpyHostToDevice));

	fill_array_rand_float(rolling_means, n_filters, 0.0F, 1.0F);
	float* d_rolling_means = 0;
	CHECK_CUDA(cudaMalloc(&d_rolling_means, n_filters * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(d_rolling_means, rolling_means, n_filters * sizeof(float), cudaMemcpyHostToDevice));

	fill_array_rand_float(rolling_variances, n_filters, 0.0F, 1.0F);
	float* d_rolling_variances = 0;
	CHECK_CUDA(cudaMalloc(&d_rolling_variances, n_filters * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(d_rolling_variances, rolling_variances, n_filters * sizeof(float), cudaMemcpyHostToDevice));

	int grid_size = n_filters;
	int block_size = BLOCKSIZE;
	int select = 2;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	if (select == 1) {
		forward_batchnorm_kernel_no_shuffle KARGS(grid_size, block_size) (d_gammas, d_betas, d_means, d_variances, d_rolling_means, d_rolling_variances, d_Z, d_Z_norm, d_act_inputs, spatial, batch_size, out_n);
	}
	else {
		forward_batchnorm_kernel KARGS(grid_size, block_size) (d_gammas, d_betas, d_means, d_variances, d_rolling_means, d_rolling_variances, d_Z, d_Z_norm, d_act_inputs, spatial, batch_size, out_n);
	}

	CHECK_CUDA(cudaGetLastError());

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("batchnorm kernel execution time: %f ms\n", milliseconds);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaMemcpy(act_inputs, d_act_inputs, out_n * batch_size * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(d_Z));
	CHECK_CUDA(cudaMemcpy(Z_norm, d_Z_norm, out_n * batch_size * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(d_Z_norm));
	CHECK_CUDA(cudaMemcpy(means, d_means, n_filters * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(d_means));
	CHECK_CUDA(cudaMemcpy(variances, d_variances, n_filters * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(d_variances));
	CHECK_CUDA(cudaMemcpy(gammas, d_gammas, n_filters * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(d_gammas));
	CHECK_CUDA(cudaMemcpy(betas, d_betas, n_filters * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(d_betas));
	CHECK_CUDA(cudaMemcpy(rolling_means, d_rolling_means, n_filters * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(d_rolling_means));
	CHECK_CUDA(cudaMemcpy(rolling_variances, d_rolling_variances, n_filters * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(d_rolling_variances));

	layer l = { 0 };
	l.out_w = (size_t)w;
	l.out_h = (size_t)h;
	l.n_filters = (size_t)n_filters;
	l.out_n = (size_t)out_n;
	l.Z = Z;
	l.Z_norm = Z_norm;
	l.act_inputs = (float*)xcalloc(out_n * batch_size, sizeof(float));
	l.means = means;
	l.variances = variances;
	l.gammas = gammas;
	l.biases = betas;
	l.rolling_means = rolling_means;
	l.rolling_variances = rolling_variances;
	forward_batchnorm(&l, (size_t)batch_size);

	float epsilon = 1e-2f;
	printf("Verifiying......\n");
	for (size_t i = 0; i < out_n * batch_size; i++) {
		//printf("%f : %f\n", l.act_inputs[i], act_inputs[i]);
		if (fabs(l.act_inputs[i] - act_inputs[i]) > epsilon || isnan(l.act_inputs[i]) || isnan(act_inputs[i])) {
			printf("Verification Failed: i = %zu, (cpu)%f != (gpu)%f\n", i, l.act_inputs[i], act_inputs[i]);
			wait_for_key_then_exit();
		}
	}
	printf("Verifiction Success!!!\n");
}

__global__ void backward_batchnorm_kernel(
	float* grads,
	float* Z, float* Z_norm,
	float* means, float* variances,
	float* gammas, float* gamma_grads,
	int spatial, int n, int batch_size)
	{

	__shared__ float warp_sums[BLOCKSIZE >> 5];
	__shared__ float gamma;
	__shared__ float mean;
	__shared__ float variance;
	__shared__ float mean_grad;
	__shared__ float variance_grad;

	int tid = threadIdx.x;
	int filter = blockIdx.x;

	int lane = threadIdx.x & 31;
	int warp_id = threadIdx.x >> 5;

	int fs = filter * spatial;

	// --- GAMMA_GRADS ---
	float thread_sum = 0.0F;
	for (int b = 0; b < batch_size; b++) {
		int offset = b * n + fs;
		for (int s = tid; s < spatial; s += BLOCKSIZE) {
			int i = offset + s;
			thread_sum += grads[i] * Z_norm[i];
			Z_norm[i] = 0.0F;
		}
	}

	for (int offset = 16; offset > 0; offset >>= 1) {
		thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
	}

	if (lane == 0) warp_sums[warp_id] = thread_sum;
	if (tid == 0) {
		gamma = gammas[filter];
		mean = means[filter];
		variance = variances[filter];
	}
	__syncthreads();

	if (warp_id == 0) {
		float block_sum = (tid < BLOCKSIZE >> 5) ? warp_sums[tid] : 0.0F;
		for (int offset = 16; offset > 0; offset >>= 1) {
			block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
		}
		if (tid == 0) {
			gamma_grads[filter] = block_sum;
		}
	}
	
	// --- MEAN_GRADS ---
	thread_sum = 0.0F;
	for (int b = 0; b < batch_size; b++) {
		int offset = b * n + fs;
		for (int s = tid; s < spatial; s += BLOCKSIZE) {
			float grad = grads[offset + s] * gamma;
			grads[offset + s] = grad;
			thread_sum += grad;
		}
	}

	for (int offset = 16; offset > 0; offset >>= 1) {
		thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
	}

	if (lane == 0) warp_sums[warp_id] = thread_sum;
	__syncthreads();

	if (warp_id == 0) {
		float block_sum = (tid < BLOCKSIZE >> 5) ? warp_sums[tid] : 0.0F;
		for (int offset = 16; offset > 0; offset >>= 1) {
			block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
		}
		if (tid == 0) {
			mean_grad = block_sum * (-1.0F / sqrtf(variance + 0.00001F));
		}
	}
	__syncthreads();

	// --- VARIANCE_GRADS ---
	thread_sum = 0.0F;
	for (int b = 0; b < batch_size; b++) {
		int offset = b * n + fs;
		for (int s = tid; s < spatial; s += BLOCKSIZE) {
			int i = offset + s;
			thread_sum += grads[i] * (Z[i] - mean);
		}
	}

	for (int offset = 16; offset > 0; offset >>= 1) {
		thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
	}

	if (lane == 0) warp_sums[warp_id] = thread_sum;
	__syncthreads();

	if (warp_id == 0) {
		float block_sum = (tid < BLOCKSIZE >> 5) ? warp_sums[tid] : 0.0F;
		for (int offset = 16; offset > 0; offset >>= 1) {
			block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
		}
		if (tid == 0) {
			variance_grad = block_sum * -0.5F * powf(variance + 0.00001F, (float)(-3.0F / 2.0F));
		}
	}
	__syncthreads();

	// --- GRADS ---
	float sb = (float)(spatial * batch_size);
	for (int b = 0; b < batch_size; b++) {
		int offset = b * n + fs;
		for (int s = tid; s < spatial; s += BLOCKSIZE) {
			int i = offset + s;
			float grad = grads[i];
			grads[i] = grad * 1.0F / sqrtf(variance + 0.00001F) + variance_grad * 2.0F * (grad - mean) / sb + mean_grad / sb;
		}
	}
}

void backward_batchnorm_gpu(float* grads,
	float* Z, float* Z_norm,
	float* means, float* variances,
	float* gammas, float* gamma_grads,
	int spatial, int n_filters, int batch_size) {

	int n = spatial * n_filters;
	int grid_size = n_filters;
	backward_batchnorm_kernel KARGS(grid_size, BLOCKSIZE) (grads, Z, Z_norm, means, variances, gammas, gamma_grads, spatial, n, batch_size);
	CHECK_CUDA(cudaPeekAtLastError());
}

void test_backward_batchnorm_gpu(void) {
	int batch_size = 8;
	int w = 320;
	int h = 320;
	int spatial = w * h;
	int n_filters = 64;
	int out_n = spatial * n_filters;
	float* Z = (float*)xcalloc(out_n * batch_size, sizeof(float));
	float* Z_norm = (float*)xcalloc(out_n * batch_size, sizeof(float));
	float* means = (float*)xcalloc(n_filters, sizeof(float));
	float* variances = (float*)xcalloc(n_filters, sizeof(float));
	float* gammas = (float*)xcalloc(n_filters, sizeof(float));
	float* gamma_grads = (float*)xcalloc(n_filters, sizeof(float));
	float* grads = (float*)xcalloc(out_n * batch_size, sizeof(float));
	
	fill_array_rand_float(Z, out_n * batch_size, 0.0F, 0.5F);
	float* d_Z = 0;
	CHECK_CUDA(cudaMalloc(&d_Z, out_n * batch_size * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(d_Z, Z, out_n * batch_size * sizeof(float), cudaMemcpyHostToDevice));

	fill_array_rand_float(Z_norm, out_n * batch_size, 0.0F, 0.1F);
	float* d_Z_norm = 0;
	CHECK_CUDA(cudaMalloc(&d_Z_norm, out_n * batch_size * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(d_Z_norm, Z_norm, out_n * batch_size * sizeof(float), cudaMemcpyHostToDevice));

	fill_array_rand_float(means, n_filters, 0.0F, 0.1F);
	float* d_means = 0;
	CHECK_CUDA(cudaMalloc(&d_means, n_filters * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(d_means, means, n_filters * sizeof(float), cudaMemcpyHostToDevice));

	fill_array_rand_float(variances, n_filters, 0.0F, 0.1F);
	for (int i = 0; i < n_filters; ++i) variances[i] = fabsf(variances[i]);
	float* d_variances = 0;
	CHECK_CUDA(cudaMalloc(&d_variances, n_filters * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(d_variances, variances, n_filters * sizeof(float), cudaMemcpyHostToDevice));

	fill_array_rand_float(gammas, n_filters, 0.0F, 0.5F);
	float* d_gammas = 0;
	CHECK_CUDA(cudaMalloc(&d_gammas, n_filters * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(d_gammas, gammas, n_filters * sizeof(float), cudaMemcpyHostToDevice));

	float* d_gamma_grads = 0;
	CHECK_CUDA(cudaMalloc(&d_gamma_grads, n_filters * sizeof(float)));

	fill_array_rand_float(grads, out_n * batch_size, 0.0F, 0.01F);
	float* d_grads = 0;
	CHECK_CUDA(cudaMalloc(&d_grads, out_n * batch_size * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(d_grads, grads, out_n * batch_size * sizeof(float), cudaMemcpyHostToDevice));

	int grid_size = n_filters;
	int block_size = BLOCKSIZE;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	backward_batchnorm_kernel KARGS(grid_size, block_size) (d_grads, d_Z, d_Z_norm, d_means, d_variances, d_gammas, d_gamma_grads, spatial, out_n, batch_size);
	
	CHECK_CUDA(cudaGetLastError());

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("batchnorm kernel execution time: %f ms\n", milliseconds);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());

	float* gpu_grads = (float*)xmalloc(out_n * batch_size * sizeof(float));
	CHECK_CUDA(cudaMemcpy(gpu_grads, d_grads, out_n * batch_size * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(d_grads));
	CHECK_CUDA(cudaMemcpy(gamma_grads, d_gamma_grads, n_filters * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(d_gamma_grads));

	CHECK_CUDA(cudaFree(d_Z));
	CHECK_CUDA(cudaFree(d_Z_norm));
	CHECK_CUDA(cudaFree(d_means));
	CHECK_CUDA(cudaFree(d_variances));
	CHECK_CUDA(cudaFree(d_gammas));

	layer l = { 0 };

	l.out_w = (size_t)w;
	l.out_h = (size_t)h;
	l.n_filters = (size_t)n_filters;
	l.out_n = (size_t)out_n;
	l.Z = Z;
	l.Z_norm = Z_norm;
	l.means = means;
	l.variances = variances;
	l.gammas = gammas;
	l.grads = grads;
	l.gamma_grads = (float*)xmalloc(n_filters * sizeof(float));
	
	backward_batchnorm(&l, (size_t)batch_size);

	float epsilon = 2e-2f;
	printf("Verifiying grads......\n");
	for (size_t i = 0; i < out_n * batch_size; i++) {
		//printf("%f : %f\n", l.grads[i], gpu_grads[i]);
		if (fabsf(l.grads[i] - gpu_grads[i]) > epsilon || isnan(l.grads[i]) || isnan(gpu_grads[i])) {
			printf("Verification Failed: i = %zu, (cpu)%f != (gpu)%f\n", i, l.grads[i], gpu_grads[i]);
			wait_for_key_then_exit();
		}
	}
	printf("Grads verification success.\n");
	printf("Verifiying gamma grads......\n");
	for (size_t i = 0; i < n_filters; i++) {
		//printf("%f : %f\n", l.gamma_grads[i], gamma_grads[i]);
		if (fabsf(l.gamma_grads[i] - gamma_grads[i]) > epsilon || isnan(l.gamma_grads[i]) || isnan(gamma_grads[i])) {
			printf("Verification Failed: i = %zu, (cpu)%f != (gpu)%f\n", i, l.gamma_grads[i], gamma_grads[i]);
			wait_for_key_then_exit();
		}
	}
	printf("Gamma grads verification success.\n");
}