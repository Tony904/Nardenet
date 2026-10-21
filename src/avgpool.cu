#ifdef GPU

#include <stdio.h>
#include <math.h>
#include "xcuda.h"
#include "utils.h"


#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#define __shfl_down_sync(...) ( __VA_ARGS__ )
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif



// GLOBAL FORWARD
__global__ void forward_avgpool_global_kernel(float* input, float* output, int spatial) {

	__shared__ float shared[BLOCKSIZE];

	int tid = threadIdx.x;
	int channel = blockIdx.x;
	int lane = threadIdx.x & 31;
	int warp_id = threadIdx.x >> 5;

	float thread_sum = 0.0F;
	int offset = channel * spatial;
	for (int s = tid; s < spatial; s += BLOCKSIZE) {
		if (s < spatial) thread_sum += input[offset + s];
	}

	for (int offset = 16; offset > 0; offset >>= 1) {
		thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
	}

	if (lane == 0) shared[warp_id] = thread_sum;
	__syncthreads();

	float sum = 0.0F;
	if (warp_id == 0) {
		sum = (tid < (BLOCKSIZE >> 5)) ? shared[tid] : 0.0F;
		for (int offset = 16; offset > 0; offset >>= 1) {
			sum += __shfl_down_sync(0xffffffff, sum, offset);
		}
		if (tid == 0) {
			output[channel] = sum / (float)spatial;
		}
	}
}
void launch_forward_avgpool_global_kernel(float* input, float* output, int spatial, int c, int batch_size) {
	forward_avgpool_global_kernel KARGS(c * batch_size, BLOCKSIZE) (input, output, spatial);
	CHECK_CUDA(cudaPeekAtLastError());
}


// GLOBAL BACKWARD
__global__ void backward_avgpool_global_kernel(float* grads_x, float* grads_y, int spatial) {
	int tid = threadIdx.x;
	int channel = blockIdx.x;
	int offset = channel * spatial;
	float grad = grads_y[channel] / (float)spatial;
	for (int s = tid; s < spatial; s += BLOCKSIZE) {
		grads_x[offset + s] += grad;
	}
}
void launch_backward_avgpool_global_kernel(float* grads_x, float* grads_y, int spatial, int c, int batch_size) {
	backward_avgpool_global_kernel KARGS(c * batch_size, BLOCKSIZE) (grads_x, grads_y, spatial);
	CHECK_CUDA(cudaPeekAtLastError());
}



// LOCAL FORWARD
__global__ void forward_avgpool_local_general_kernel(float* src, float* dst, int src_w, int src_h, int dst_w, int dst_sp, int dst_n, int ksize, int stride) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < dst_n) {
		int ch = gtid / (dst_w * dst_sp);
		int s = gtid % (dst_w * dst_sp);
		int dst_row = s / dst_w;
		int dst_col = s % dst_w;
		int src_row = dst_row * stride;
		int src_col = dst_col * stride;
		int src_index = ch * (src_w * src_h) + src_w * src_row + src_col;

		float sum = 0.0F;
		for (int krow = 0; krow < ksize; krow++) {
			int offset = src_index + krow * src_w;
			for (int kcol = 0; kcol < ksize; kcol++) {
				sum += src[offset + kcol];
			}
		}
		dst[gtid] = sum / (ksize * ksize);
	}
}
void launch_forward_avgpool_local_general_kernel(float* src, float* dst, int src_w, int src_h, int dst_w, int dst_h, int dst_n, int ksize, int stride) {
	int grid_size = GET_GRIDSIZE(dst_n, BLOCKSIZE);
	forward_avgpool_local_general_kernel KARGS(grid_size, BLOCKSIZE) (src, dst, src_w, src_h, dst_w, dst_w * dst_h, dst_n, ksize, stride);
	CHECK_CUDA(cudaPeekAtLastError());
}



// ksize = 2, stride = 2
__global__ void forward_avgpool_local_standard_kernel(float* src, float* dst, int src_w, int src_h, int dst_w, int dst_h, int dst_n) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < dst_n) {
		int ch = gtid / (dst_w * dst_h);
		int s = gtid % (dst_w * dst_h);
		int dst_row = s / dst_w;
		int dst_col = s % dst_w;
		int src_row = dst_row * 2;
		int src_col = dst_col * 2;
		int i = ch * (src_w * src_h) + src_w * src_row + src_col;

		dst[gtid] = (src[i] + src[i + 1] + src[i + src_w] + src[i + src_w + 1]) / 4.0F;
	}
}
void launch_forward_avgpool_local_standard_kernel(float* src, float* dst, int src_w, int src_h, int dst_w, int dst_h, int dst_n) {
	int grid_size = GET_GRIDSIZE(dst_n, BLOCKSIZE);
	forward_avgpool_local_standard_kernel KARGS(grid_size, BLOCKSIZE) (src, dst, src_w, src_h, dst_w, dst_h, dst_n);
	CHECK_CUDA(cudaPeekAtLastError());
}



// LOCAL BACKWARD
__global__ void backward_avgpool_local_general_kernel(float* grads_x, float* grads_y, int src_w, int src_h, int dst_w, int dst_sp, int dst_n, int ksize, int stride) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < dst_n) {
		int ch = gtid / dst_sp;
		int s = gtid % dst_sp;
		int dst_row = s / dst_w;
		int dst_col = s % dst_w;
		int src_row = dst_row * stride;
		int src_col = dst_col * stride;
		int src_index = ch * (src_w * src_h) + src_w * src_row + src_col;

		float grad = grads_y[gtid] / (ksize * ksize);
		for (int krow = 0; krow < ksize; krow++) {
			int offset = src_index + krow * src_w;
			for (int kcol = 0; kcol < ksize; kcol++) {
				grads_x[offset + kcol] += grad;
			}
		}
	}
}
void launch_backward_avgpool_local_general_kernel(float* grads_x, float* grads_y, int src_w, int src_h, int dst_w, int dst_h, int dst_n, int ksize, int stride) {
	int grid_size = GET_GRIDSIZE(dst_n, BLOCKSIZE);
	backward_avgpool_local_general_kernel KARGS(grid_size, BLOCKSIZE) (grads_x, grads_y, src_w, src_h, dst_w, dst_w * dst_h, dst_n, ksize, stride);
	CHECK_CUDA(cudaPeekAtLastError());
}



// ksize = 2, stride = 2
__global__ void backward_avgpool_local_standard_kernel(float* grads_x, float* grads_y, int src_w, int src_h, int dst_w, int dst_sp, int dst_n) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < dst_n) {
		int ch = gtid / dst_sp;
		int s = gtid % dst_sp;
		int dst_row = s / dst_w;
		int dst_col = s % dst_w;
		int src_row = dst_row * 2;
		int src_col = dst_col * 2;
		int src_index = ch * (src_w * src_h) + src_w * src_row + src_col;

		float grad = grads_y[gtid] / 4.0F;
		grads_x[src_index] += grad;
		grads_x[src_index + 1] += grad;
		grads_x[src_index + src_w] += grad;
		grads_x[src_index + src_w + 1] += grad;
	}
}
void launch_backward_avgpool_local_standard_kernel(float* grads_x, float* grads_y, int src_w, int src_h, int dst_w, int dst_h, int dst_n) {
	int grid_size = GET_GRIDSIZE(dst_n, BLOCKSIZE);
	backward_avgpool_local_standard_kernel KARGS(grid_size, BLOCKSIZE) (grads_x, grads_y, src_w, src_h, dst_w, dst_w * dst_h, dst_n);
	CHECK_CUDA(cudaPeekAtLastError());
}

#endif