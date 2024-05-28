#ifndef XARRAYS_H
#define XARRAYS_H


#include <stdlib.h>


#ifdef __cplusplus
extern "C" {
#endif

	typedef struct floatarr floatarr;
	typedef struct intarr intarr;

	void free_floatarr(floatarr* p);
	void print_floatarr(floatarr* p);
	void free_intarr(intarr* p);
	void print_intarr(intarr* p);

	typedef struct floatarr {
		size_t n;  // length of array
		float* a;
	} floatarr;

	typedef struct intarr {
		size_t n;  // length of array
		int* a;
	} intarr;

#ifdef __cplusplus
}
#endif
#endif