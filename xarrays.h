#ifndef XARRAYS_H
#define XARRAYS_H

#include <stdlib.h>


#ifdef __cplusplus
extern "C" {
#endif


	typedef struct floatarr floatarr;
	typedef struct intarr intarr;

	typedef struct floatarr {
		size_t length;
		float* vals;
	} floatarr;

	typedef struct intarr {
		size_t length;
		int* vals;
	} intarr;


#ifdef __cplusplus
}
#endif
#endif