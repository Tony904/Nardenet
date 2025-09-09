#ifdef GPU

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "xcuda.h"
#include "xallocs.h"
#include "utils.h"
#include "activations.h"
#include "blas.h"


#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#define __shfl_down_sync(...) ( __VA_ARGS__ )
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif


__device__ __forceinline__ float sigmoid_x_kernel(float x) { return 1.0F / (1.0F + expf(-x)); }
__device__ __forceinline__ float tanh_x_kernel(float x) { return (2.0F / (1.0F + expf(-2.0F * x)) - 1.0F); }
__device__ __forceinline__ float relu_x_kernel(float x) { return x * (x > 0.0F); }
__device__ __forceinline__ float leaky_x_kernel(float x) { return x > 0.0F ? x : 0.1F * x; }
__device__ __forceinline__ float softplus_x_kernel(float x, float t) { return (x > t) ? x : (x < -t) ? expf(x) : logf(expf(x) + 1.0F); }
__device__ __forceinline__ float mish_x_kernel(float x, float thresh) { return x * tanh_x_kernel(softplus_x_kernel(x, thresh)); }



__global__ void activate_relu_kernel(float* Z, float* output, int n) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < n) output[gtid] = relu_x_kernel(Z[gtid]);
}
void activate_relu_gpu(float* Z, float* output, size_t out_n, size_t batch_size) {
	int n = (int)(out_n * batch_size);
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	activate_relu_kernel KARGS(grid_size, BLOCKSIZE) (Z, output, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void activate_leaky_relu_kernel(float* Z, float* output, int n) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < n) output[gtid] = leaky_x_kernel(Z[gtid]);
}
void activate_leaky_relu_gpu(float* Z, float* output, size_t out_n, size_t batch_size) {
	int n = (int)(out_n * batch_size);
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	activate_leaky_relu_kernel KARGS(grid_size, BLOCKSIZE) (Z, output, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void activate_mish_kernel(float* Z, float* output, int n) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < n) output[gtid] = mish_x_kernel(Z[gtid], MISH_THRESH);
}
void activate_mish_gpu(float* Z, float* output, size_t out_n, size_t batch_size) {
	int n = (int)(out_n * batch_size);
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	activate_mish_kernel KARGS(grid_size, BLOCKSIZE) (Z, output, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void activate_sigmoid_kernel(float* Z, float* output, int n) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < n) output[gtid] = sigmoid_x_kernel(Z[gtid]);
}
void activate_sigmoid_gpu(float* Z, float* output, size_t out_n, size_t batch_size) {
	int n = (int)(out_n * batch_size);
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	activate_sigmoid_kernel KARGS(grid_size, BLOCKSIZE) (Z, output, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void activate_tanh_kernel(float* Z, float* output, int n) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < n) output[gtid] = tanh_x_kernel(Z[gtid]);
}
void activate_tanh_gpu(float* Z, float* output, size_t out_n, size_t batch_size) {
	int n = (int)(out_n * batch_size);
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	activate_sigmoid_kernel KARGS(grid_size, BLOCKSIZE) (Z, output, n);
	CHECK_CUDA(cudaPeekAtLastError());
}


// Each block handles one batch
__global__ void activate_softmax_kernel(float* Z, float* output, int out_n) {

	__shared__ float shared[BLOCKSIZE >> 5];

	int blocksize = blockDim.x;
	int tid = threadIdx.x;
	int lane = threadIdx.x & 31;
	int warp_id = threadIdx.x >> 5;
	int index = out_n * blockIdx.x + tid;

	float z = (tid < out_n) ? Z[index] : -FLT_MAX;
	float max_val = z;

	for (int offset = 16; offset > 0; offset >>= 1) {
		max_val = fmaxf(__shfl_down_sync(0xffffffff, max_val, offset), max_val);
	}
	if (lane == 0) shared[warp_id] = max_val;
	__syncthreads();

	if (warp_id == 0) {
		max_val = (tid < blocksize >> 5) ? shared[tid] : -FLT_MAX;
		for (int offset = 16; offset > 0; offset >>= 1) {
			max_val = fmaxf(__shfl_down_sync(0xffffffff, max_val, offset), max_val);
		}
		if (tid == 0) shared[0] = max_val;
	}
	__syncthreads();

	max_val = shared[0];
	z = (tid < out_n) ? expf(z - max_val) : 0.0F;

	float sum = z;
	for (int offset = 16; offset > 0; offset >>= 1) {
		sum += __shfl_down_sync(0xffffffff, sum, offset);
	}
	if (lane == 0) shared[warp_id] = sum;
	__syncthreads();

	if (warp_id == 0) {
		sum = (tid < blocksize >> 5) ? shared[tid] : 0.0F;
		for (int offset = 16; offset > 0; offset >>= 1) {
			sum += __shfl_down_sync(0xffffffff, sum, offset);
		}
		if (tid == 0) shared[0] = sum;
	}
	__syncthreads();

	z /= shared[0];
	if (tid < out_n) output[index] = z;
}

void activate_softmax_gpu(float* Z, float* output, size_t out_n, size_t batch_size) {
	activate_softmax_kernel KARGS((int)batch_size, BLOCKSIZE) (Z, output, (int)out_n);
	CHECK_CUDA(cudaPeekAtLastError());
}

void test_activate_softmax_gpu(void) {
	size_t out_n = 256;
	size_t batch_size = 64;
	float* Z = (float*)xmalloc(out_n * batch_size * sizeof(float));
	float* output = (float*)xmalloc(out_n * batch_size * sizeof(float));

	fill_array_rand_float(Z, out_n * batch_size, 0.0, 1.0);

	float* d_Z = 0;
	float* d_output = 0;

	CHECK_CUDA(cudaMalloc(&d_Z, out_n * batch_size * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_output, out_n * batch_size * sizeof(float)));

	CHECK_CUDA(cudaMemcpy(d_Z, Z, out_n * batch_size * sizeof(float), cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);


	activate_softmax_gpu(d_Z, d_output, out_n, batch_size);


	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Kernel execution time: %f ms\n", milliseconds);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	CHECK_CUDA(cudaDeviceSynchronize());

	float* output_gpu = (float*)xmalloc(out_n * batch_size * sizeof(float));
	CHECK_CUDA(cudaMemcpy(output_gpu, d_output, out_n * batch_size * sizeof(float), cudaMemcpyDeviceToHost));

	CHECK_CUDA(cudaFree(d_Z));
	CHECK_CUDA(cudaFree(d_output));

	activate_softmax(Z, output, out_n, batch_size);

	float epsilon = 1e-5f;
	printf("Verifiying......\n");
	for (size_t i = 0; i < out_n * batch_size; i++) {
		//printf("%f : %f\n", output[i], output_gpu[i]);
		if (fabs(output[i] - output_gpu[i]) > epsilon || isnan(output[i]) || isnan(output_gpu[i])) {
			printf("Verification Failed: i = %zu, (cpu)%f != (gpu)%f\n", i, output[i], output_gpu[i]);
			wait_for_key_then_exit();
		}
	}
	printf("Verifiction Success!!!\n");
}

#endif