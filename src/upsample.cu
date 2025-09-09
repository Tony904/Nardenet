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


__global__ void forward_upsample_kernel(float* input, float* output, int spatial, int width, int ksize) {
	int tid = threadIdx.x;
	int channel = blockIdx.x;
	int out_width = width * ksize;
	int offset = channel * spatial;
	int out_offset = offset * ksize * ksize;

	for (int s = tid; s < spatial; s += BLOCKSIZE) {
		float val = input[offset + s];
		int row = s / width;
		int col = s - (row * width);
		for (int krow = 0; krow < ksize; krow++) {
			int out_row = row * ksize + krow;
			for (int kcol = 0; kcol < ksize; kcol++) {
				int out_col = col * ksize + kcol;
				output[out_offset + out_row * out_width + out_col] = val;
			}
		}
	}
	
}
void launch_forward_upsample_kernel(float* input, float* output, int w, int h, int c, int ksize, int batch_size) {
	int n = w * h * c * batch_size;
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	forward_upsample_kernel KARGS(grid_size, BLOCKSIZE) (input, output, w * h, w, ksize);
	CHECK_CUDA(cudaPeekAtLastError());
}

// spatial and width is of grads_x
__global__ void backward_upsample_kernel(float* grads_x, float* grads_y, int spatial, int width, int ksize) {
	int tid = threadIdx.x;
	int channel = blockIdx.x;
	int out_width = width * ksize;
	int offset = channel * spatial;
	int out_offset = offset * ksize * ksize;

	for (int s = tid; s < spatial; s += BLOCKSIZE) {
		int row = s / width;
		int col = s - (row * width);
		for (int krow = 0; krow < ksize; krow++) {
			int out_row = row * ksize + krow;
			for (int kcol = 0; kcol < ksize; kcol++) {
				int out_col = col * ksize + kcol;
				float val = grads_y[out_offset + out_row * out_width + out_col];
				grads_x[offset + s] += val;
			}
		}
	}
}
void launch_backward_upsample_kernel(float* grads_x, float* grads_y, int w, int h, int c, int ksize, int batch_size) {
	int n = w * h * c * batch_size;
	int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
	forward_upsample_kernel KARGS(grid_size, BLOCKSIZE) (grads_x, grads_y, w * h, w, ksize);
	CHECK_CUDA(cudaPeekAtLastError());
}

#endif