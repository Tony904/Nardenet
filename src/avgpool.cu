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

// GLOBAL
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

// LOCAL
//__global__ void forward_avgpool_local_kernel(float* input) {
//
//}
//
//void launch_forward_avgpool_local_kernel(float* input) {
//	for (size_t ch = 0; ch < inl_c; ch++) {
//		float* input_ch = &inl_output[ch * wh];
//		float* output_ch = &l_output[ch * out_wh];
//		for (int krow = 0; krow < ksize; krow++) {
//			for (int kcol = 0; kcol < ksize; kcol++) {
//				float* output = output_ch;
//				int in_row = krow;
//				for (size_t out_rows = out_h; out_rows; out_rows--) {
//					int in_col = kcol;
//					int r = in_row * w;
//					for (size_t out_cols = out_w; out_cols; out_cols--) {
//						*output += input_ch[r + in_col];
//						output++;
//						in_col += stride;
//					}
//					in_row += stride;
//				}
//			}
//		}
//		for (size_t s = 0; s < out_wh; s++) {
//			output_ch[s] /= divisor;
//		}
//	}
//}

#endif