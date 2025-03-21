#include "xcuda.h"
#include <stdio.h>
#include "utils.h"



void ___check_cuda(cudaError_t x, const char* const filename, const char* const funcname, const int line, const char* time) {
	if (x != cudaSuccess) {
		fprintf(stderr, "CUDA error %s\ntime: %s", cudaGetErrorString(x), time);
		print_location(filename, funcname, line);
		wait_for_key_then_exit();
	}
}
