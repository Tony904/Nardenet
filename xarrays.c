#include "xarrays.h"
#include "xallocs.h"
#include <assert.h>
#include <stdio.h>


void free_floatarr(floatarr* p) {
	xfree(p->vals);
	xfree(p);
}

void free_intarr(intarr* p) {
	xfree(p->vals);
	xfree(p);
}

void print_floatarr(floatarr* p) {
	size_t n = p->length;
	assert(n > 0);
	size_t i;
	for (i = 0; i < n - 1; i++) {
		printf("%f, ", p->vals[i]);
	}
	printf("%f\n", p->vals[i]);
}

void print_intarr(intarr* p) {
	size_t n = p->length;
	assert(n > 0);
	size_t i;
	for (i = 0; i < n - 1; i++) {
		printf("%d, ", p->vals[i]);
	}
	printf("%d\n", p->vals[i]);
}
