#include <stdio.h>
#include <math.h>
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
		errors[index] = (t) ? -logf(p) : 0.0F;
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
		errors[index] = -t * logf(p) - (1.0F - t) * logf(1.0F - p);
	}
}
void launch_loss_bce_kernel(float* grads, float* output, float* truth, float* errors, int n, int batch_size) {
	n = n * batch_size;
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	loss_cce_kernel KARGS(grid_size, BLOCKSIZE) (grads, output, truth, errors, n);
	CHECK_CUDA(cudaPeekAtLastError());
}