#ifndef XCUDA_H
#define XCUDA_H


#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "locmacro.h"


#ifndef __TIME__
#define __TIME__
#endif

#ifdef __cplusplus
extern "C" {
#endif


	void ___check_cuda(cudaError_t x, const char* const filename, const char* const funcname, const int line, const char* time);

#define CHECK_CUDA(x) ___check_cuda(x, NARDENET_LOCATION, " - " __TIME__);

	/*void test_cuda(void);*/
	void cuda_test_im2col(void);
	void cuda_test_im2col_shared(void);
	void print_gpu_props(void);


#ifdef __cplusplus
}
#endif
#endif