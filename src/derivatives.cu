#include <stdio.h>
#include "xcuda.h"
#include "xallocs.h"
#include "utils.h"
#include <math.h>
#include <float.h>
#include "network.h"
#include "activations.h"


#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#define __shfl_down_sync(...) ( __VA_ARGS__ )
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif

#define BLOCKSIZE 512


__device__ __forceinline__ float softplus_x_kernel(float x, float t) { return (x > t) ? x : (x < -t) ? expf(x) : logf(expf(x) + 1.0F); }
__device__ __forceinline__ float tanh_x_kernel(float x) { return (2.0F / (1.0F + expf(-2.0F * x)) - 1.0F); }


__global__ void grads_sigmoid_kernel(float* grads, float* act_output, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float x = act_output[i];
	if (i < n) grads[i] *= x * (1.0F - x);
}
void get_grads_sigmoid_gpu(float* grads, float* act_output, size_t out_n, size_t batch_size) {
	int n = (int)(out_n * batch_size);
	int block_size = BLOCKSIZE;
	int grid_size = ((n / block_size) + (((n % block_size) > 0) ? 1 : 0));
	grads_sigmoid_kernel KARGS(grid_size, block_size) (grads, act_output, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void grads_mish_kernel(float* grads, float* act_inputs, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float inp = act_inputs[i];
	float sp = softplus_x_kernel(inp, MISH_THRESH);
	float grad_sp = 1.0F - expf(-sp);
	float tsp = tanh_x_kernel(sp);
	float grad_tsp = (1.0F - tsp * tsp) * grad_sp;
	float grad = inp * grad_tsp + tsp;
	grads[i] += grad;
}
void get_grads_mish_gpu(float* grads, float* act_inputs, size_t out_n, size_t batch_size) {
	int n = (int)(out_n * batch_size);
	int block_size = BLOCKSIZE;
	int grid_size = ((n / block_size) + (((n % block_size) > 0) ? 1 : 0));
	grads_mish_kernel KARGS(grid_size, block_size) (grads, act_inputs, n);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void grads_relu_kernel(float* grads, float* act_inputs, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float x = act_inputs[i];
	if (i < n) grads[i] *= (act_inputs[i] > 0);
}
void get_grads_relu_gpu(float* grads, float* act_inputs, size_t out_n, size_t batch_size) {
	size_t n = out_n * batch_size;
	int n = (int)(out_n * batch_size);
	int block_size = BLOCKSIZE;
	int grid_size = ((n / block_size) + (((n % block_size) > 0) ? 1 : 0));
	grads_relu_kernel KARGS(grid_size, block_size) (grads, act_inputs, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


void get_grads_leaky_relu_gpu(float* grads, float* act_inputs, size_t out_n, size_t batch_size) {
	size_t n = out_n * batch_size;
	size_t i;
	for (i = 0; i < n; i++) {
		grads[i] *= (act_inputs[i] > 0.0F) ? 1.0F : 0.1F;
	}
}


void get_grads_tanh_gpu(float* grads, float* act_inputs, size_t out_n, size_t batch_size) {
	size_t n = out_n * batch_size;
	size_t i;
	for (i = 0; i < n; i++) {
		float x = tanh_x(act_inputs[i]);
		grads[i] *= 1 - (x * x);
	}
}


void regularize_l1_gpu(float* weight_grads, float* weights, size_t size, float decay) {
	size_t i;
	for (i = 0; i < size; i++) {
		weight_grads[i] -= ((weights[i] > 0.0F) ? decay : -decay);
	}
}


void regularize_l2_gpu(float* weight_grads, float* weights, size_t size, float decay) {
	size_t i;
	for (i = 0; i < size; i++) {
		weight_grads[i] -= weights[i] * decay;
	}
}