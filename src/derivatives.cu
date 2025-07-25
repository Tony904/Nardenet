#include <stdio.h>
#include "xcuda.h"
#include <math.h>
#include "activations.h"


#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#define __shfl_down_sync(...) ( __VA_ARGS__ )
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif


__device__ __forceinline__ float softplus_x_kernel(float x, float t) { return (x > t) ? x : (x < -t) ? expf(x) : logf(expf(x) + 1.0F); }
__device__ __forceinline__ float tanh_x_kernel(float x) { return (2.0F / (1.0F + expf(-2.0F * x)) - 1.0F); }



__global__ void grads_sigmoid_kernel(float* grads, float* act_output, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float x = act_output[i];
	if (i < n) grads[i] *= x * (1.0F - x);
}
void get_grads_sigmoid_gpu(float* grads, float* act_output, size_t out_n, size_t batch_size) {
	int n = (int)(out_n * batch_size);
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	grads_sigmoid_kernel KARGS(grid_size, BLOCKSIZE) (grads, act_output, n);
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
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	grads_mish_kernel KARGS(grid_size, BLOCKSIZE) (grads, act_inputs, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void grads_relu_kernel(float* grads, float* act_inputs, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float x = act_inputs[i];
	if (i < n) grads[i] *= (act_inputs[i] > 0);
}
void get_grads_relu_gpu(float* grads, float* act_inputs, size_t out_n, size_t batch_size) {
	int n = (int)(out_n * batch_size);
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	grads_relu_kernel KARGS(grid_size, BLOCKSIZE) (grads, act_inputs, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void grads_leaky_relu_kernel(float* grads, float* act_inputs, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	grads[i] *= ((act_inputs[i] > 0.0F) ? 1.0F : 0.1F);
}
void get_grads_leaky_relu_gpu(float* grads, float* act_inputs, size_t out_n, size_t batch_size) {
	int n = (int)(out_n * batch_size);
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	grads_leaky_relu_kernel KARGS(grid_size, BLOCKSIZE) (grads, act_inputs, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void grads_tanh_kernel(float* grads, float* act_inputs, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float x = tanh_x_kernel(act_inputs[i]);
	grads[i] *= 1 - (x * x);
}
void get_grads_tanh_gpu(float* grads, float* act_inputs, size_t out_n, size_t batch_size) {
	int n = (int)(out_n * batch_size);
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	grads_tanh_kernel KARGS(grid_size, BLOCKSIZE) (grads, act_inputs, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void regularize_l1_kernel(float* weight_grads, float* weights, int size, float decay) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	weight_grads[i] -= (weights[i] > 0.0F) ? decay : -decay;
}
void regularize_l1_gpu(float* weight_grads, float* weights, size_t size, float decay) {
	int n = (int)(size);
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	regularize_l1_kernel KARGS(grid_size, BLOCKSIZE) (weight_grads, weights, n, decay);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void regularize_l2_kernel(float* weight_grads, float* weights, int size, float decay) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	weight_grads[i] -= weights[i] * decay;
}
void regularize_l2_gpu(float* weight_grads, float* weights, size_t size, float decay) {
	int n = (int)(size);
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	regularize_l2_kernel KARGS(grid_size, BLOCKSIZE) (weight_grads, weights, n, decay);
	CHECK_CUDA(cudaPeekAtLastError());
}