#ifndef XCUDA_H
#define XCUDA_H


#include <stdlib.h>
#include "locmacro.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_backend.h>

#ifndef __TIME__
#define __TIME__
#endif

#ifdef __cplusplus
extern "C" {
#endif


	void ___check_cuda(cudaError_t x, const char* const filename, const char* const funcname, const int line, const char* time);
	void ___check_cudnn(cudnnStatus_t x, const char* const filename, const char* const funcname, const int line, const char* time);


#define CHECK_CUDA(x) ___check_cuda(x, NARDENET_LOCATION, " - " __TIME__);
#define CHECK_CUDNN(x) ___check_cudnn(x, NARDENET_LOCATION, " - " __TIME__);


#ifdef __cplusplus
}
#endif
#endif