#ifdef GPU

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "xcuda.h"


#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#define __shfl_down_sync(...) ( __VA_ARGS__ )
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif


// ksize = 2, stride = 2, dst_w and dst_h must be even
__global__ void forward_maxpool_even_dst_wh_k2s2_kernel(float* src, float* dst, float** max_ptrs, int src_w, int src_h, int dst_w, int dst_h, int dst_n) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;

	if (gtid < dst_n) {
		const int ksize = 2;

		int ch = gtid / (dst_w * dst_h);
		int s = gtid % (dst_w * dst_h);
		int dst_row = s / dst_w;
		int dst_col = s % dst_w;
		int src_row = dst_row * ksize;
		int src_col = dst_col * ksize;
		int src_index = ch * (src_w * src_h) + src_w * src_row + src_col;

		float max_val = src[src_index];
		int max_index = src_index;

		float tmp = src[src_index + 1];
		if (tmp > max_val) {
			max_val = tmp;
			max_index = src_index + 1;
		}

		tmp = src[src_index + src_w];
		if (tmp > max_val) {
			max_val = tmp;
			max_index = src_index + src_w;
		}

		tmp = src[src_index + src_w + 1];
		if (tmp > max_val) {
			max_val = tmp;
			max_index = src_index + src_w + 1;
		}

		dst[gtid] = max_val;
		max_ptrs[gtid] = &src[max_index];
	}
}

__global__ void forward_maxpool_general_kernel(float* src, float* dst, float* grads, float** max_ptrs, int src_w, int src_h, int dst_w, int dst_h, int dst_n, int ksize, int stride) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;

	if (gtid < dst_n) {

		int ch = gtid / (dst_w * dst_h);
		int s = gtid % (dst_w * dst_h);
		int dst_row = s / dst_w;
		int dst_col = s % dst_w;
		int src_row = dst_row * stride;
		int src_col = dst_col * stride;
		int src_index = ch * (src_w * src_h) + src_row * src_w + src_col;

		float max_val = -FLT_MAX;
		int max_index = -1;
		float val;

		for (int krow = 0; krow < ksize; krow++) {
			int offset = src_index + krow * src_w;
			for (int kcol = 0; kcol < ksize; kcol++) {
				if (src_row + krow < src_h && src_col + kcol < src_w) {
					val = src[offset + kcol];
					if (val > max_val) {
						max_val = val;
						max_index = offset + kcol;
					}
				}
			}
		}

		dst[gtid] = max_val;
		max_ptrs[gtid] = &grads[max_index];
	}
}
void launch_forward_maxpool_general_kernel(float* input, float* output, float* grads, float** max_ptrs, int src_w, int src_h, int dst_w, int dst_h, int dst_n, int ksize, int stride) {
	int grid_size = GET_GRIDSIZE(dst_n, BLOCKSIZE);
	forward_maxpool_general_kernel KARGS(grid_size, BLOCKSIZE) (input, output, grads, max_ptrs, src_w, src_h, dst_w, dst_h, dst_n, ksize, stride);
	CHECK_CUDA(cudaPeekAtLastError());
}

// ksize = 2, stride = 2, n = batch_size * dst_n
__global__ void forward_maxpool_standard_kernel(float* src, float* dst, float* grads, float** max_ptrs, int src_w, int src_h, int dst_w, int dst_h, int dst_n) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;

	if (gtid < dst_n) {
		const int ksize = 2;

		int ch = gtid / (dst_w * dst_h);
		int s = gtid % (dst_w * dst_h);
		int dst_row = s / dst_w;
		int dst_col = s % dst_w;
		int src_row = dst_row * ksize;
		int src_col = dst_col * ksize;
		int src_index = ch * (src_w * src_h) + src_w * src_row + src_col;

		float max_val = src[src_index];
		int max_index = src_index;
		float tmp;

		if (src_col < src_w) {
			tmp = src[src_index + 1];
			if (tmp > max_val) {
				max_val = tmp;
				max_index = src_index + 1;
			}

			tmp = src[src_index + src_w + 1];
			if (tmp > max_val) {
				max_val = tmp;
				max_index = src_index + src_w + 1;
			}
		}

		if (src_row < src_h) {
			tmp = src[src_index + src_w];
			if (tmp > max_val) {
				max_val = tmp;
				max_index = src_index + src_w;
			}
		}

		dst[gtid] = max_val;
		max_ptrs[gtid] = &grads[max_index];
	}
}
void launch_forward_maxpool_standard_kernel(float* src, float* dst, float* grads, float** max_ptrs, int src_w, int src_h, int dst_w, int dst_h, int dst_n) {
	int grid_size = GET_GRIDSIZE(dst_n, BLOCKSIZE);
	forward_maxpool_standard_kernel KARGS(grid_size, BLOCKSIZE) (src, dst, grads, max_ptrs, src_w, src_h, dst_w, dst_h, dst_n);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void backward_maxpool_kernel(float* grads, float** max_ptrs, int n) {
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < n) {
		*max_ptrs[gtid] += grads[gtid];
	}
}
// n = grads size
void launch_backward_maxpool_kernel(float* grads, float** max_ptrs, int n) {
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	backward_maxpool_kernel KARGS(grid_size, BLOCKSIZE) (grads, max_ptrs, n);
	CHECK_CUDA(cudaPeekAtLastError());
}

#endif