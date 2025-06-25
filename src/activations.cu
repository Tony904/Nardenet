#include <stdio.h>
#include "xcuda.h"
#include "xallocs.h"
#include "utils.h"
#include <math.h>
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

__device__ __forceinline__ float sigmoid_x_kernel(float x) { return 1.0F / (1.0F + expf(-x)); }
__device__ __forceinline__ float tanh_x_kernel(float x) { return (2.0F / (1.0F + expf(-2.0F * x)) - 1.0F); }
__device__ __forceinline__ float relu_x_kernel(float x) { return x * (x > 0.0F); }
__device__ __forceinline__ float leaky_x_kernel(float x) { return x > 0.0F ? x : 0.1F * x; }
__device__ __forceinline__ float softplus_x_kernel(float x, float t) { return (x > t) ? x : (x < -t) ? expf(x) : logf(expf(x) + 1.0F); }
__device__ __forceinline__ float mish_x_kernel(float x, float thresh) { return x * tanh_x(softplus_x(x, thresh)); }

__global__ void activate_relu_kernel(float* Z, float* output, int n) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < n) output[gtid] = relu_x_kernel(Z[gtid]);
}

__global__ void activate_leaky_relu_kernel(float* Z, float* output, int n) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < n) output[gtid] = leaky_x_kernel(Z[gtid]);
}

__global__ void activate_mish_kernel(float* Z, float* output, int n) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < n) output[gtid] = mish_x_kernel(Z[gtid], MISH_THRESH);
}

__global__ void activate_sigmoid_kernel(float* Z, float* output, int n) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < n) output[gtid] = sigmoid_x_kernel(Z[gtid]);
}

__global__ void activate_tanh_kernel(float* Z, float* output, int n) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < n) output[gtid] = tanh_x_kernel(Z[gtid]);
}

__global__ void activate_softmax_kernel(float* Z, float* output, size_t out_n, size_t batch_size) {



	size_t s;
	for (s = 0; s < batch_size; s++) {
		float* z = &Z[s * out_n];
		float* a = &output[s * out_n];
		float maxval = z[0];
		for (size_t i = 1; i < out_n; i++) {
			if (z[i] > maxval) maxval = z[i];
		}
		float sum = 0.0F;
		for (size_t i = 0; i < out_n; i++) {
			float e = expf(z[i] - maxval);
			sum += e;
			a[i] = e;
		}
		for (size_t i = 0; i < out_n; i++) a[i] /= sum;
	}
}