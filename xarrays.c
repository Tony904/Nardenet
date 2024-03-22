#include "xarrays.h"
#include "xallocs.h"
#include <assert.h>
#include <stdio.h>


void free_floatarr(floatarr* farr) {
	xfree(farr->vals);
	xfree(farr);
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