#ifdef GPU

#include <stdio.h>
#include "xcuda.h"


#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#define __shfl_down_sync(...) ( __VA_ARGS__ )
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif



__global__ void update_kernel(float* vals, float* grads, float* velocities, int n_vals, float momentum, float rate) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n_vals) {
		float v_old = velocities[index];
		float v_new = momentum * v_old - rate * grads[index];
		vals[index] += -momentum * v_old * (1 + momentum) * v_new;  // Nesterov momentum
		velocities[index] = v_new;
		grads[index] = 0.0F;
	}
}
void launch_update_kernel(float* vals, float* grads, float* velocities, int n_vals, int batch_size, float momentum, float rate) {
	int n = n_vals * batch_size;
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	update_kernel KARGS(grid_size, BLOCKSIZE) (vals, grads, velocities, n, momentum, rate);
	CHECK_CUDA(cudaPeekAtLastError());
}

#endif