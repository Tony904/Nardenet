#ifndef XCUDA_H
#define XCUDA_H


#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "locmacro.h"
#include "image.h"


#ifndef __TIME__
#define __TIME__
#endif

#ifdef __cplusplus
extern "C" {
#endif


	void ___check_cuda(cudaError_t x, const char* const filename, const char* const funcname, const int line, const char* time);

#define CHECK_CUDA(x) ___check_cuda(x, NARDENET_LOCATION, " - " __TIME__)
#define BLOCKSIZE 512
#define GET_GRIDSIZE(n, blocksize) (n / blocksize) + (((n % blocksize) > 0) ? 1 : 0)
	
	void print_gpu_props(void);

	void sum_arrays_gpu(float* A, float* B, int n);
	void copy_array_gpu(float* src, float* dst, int n);
	void scale_array_gpu(float* A, float scalar, int n);
	void clamp_array_gpu(float* A, float upper, float lower, int n);
	void transform_colorspace_gpu(image* img, float brightness_scalar, float contrast_scalar, float saturation_scalar, float hue_shift);


#ifdef __cplusplus
}
#endif
#endif