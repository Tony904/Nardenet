#ifdef GPU

#include <stdio.h>
#include "xcuda.h"
#include "xallocs.h"
#include "blas.h"
#include "utils.h"


#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#define __shfl_down_sync(...) ( __VA_ARGS__ )
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif

__global__ void update_kernel(float* vals, float* grads, float* velocities, int n_vals, double momentum, double rate) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n_vals) {
		double v_old = velocities[index];
		double v_new = momentum * v_old - rate * (double)grads[index];
		vals[index] += -momentum * v_old + (1.0 + momentum) * v_new;  // Nesterov momentum
		velocities[index] = v_new;
		grads[index] = 0.0F;
	}
}
void launch_update_kernel(float* vals, float* grads, float* velocities, int n_vals, float momentum, float rate) {
	int grid_size = GET_GRIDSIZE(n_vals, BLOCKSIZE);
	update_kernel KARGS(grid_size, BLOCKSIZE) (vals, grads, velocities, n_vals, (double)momentum, (double)rate);
	CHECK_CUDA(cudaPeekAtLastError());
}

#endif