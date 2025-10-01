#include "xarrays.h"
#include "xallocs.h"
#include <stdio.h>


void free_floatarr(floatarr* p) {
	xfree(&p->a);
	xfree(&p);
}

void free_intarr(intarr* p) {
	xfree(&p->a);
	xfree(&p);
}

void print_floatarr(floatarr* p) {
	if (!p) {
		printf("\n");
		return;
	}
	size_t n = p->n;
	if (n < 1) {
		printf("none\n");
		return;
	}
	size_t i;
	for (i = 0; i < n - 1; i++) {
		printf("%f, ", p->a[i]);
	}
	printf("%f\n", p->a[i]);
}

void print_intarr(intarr* p) {
	if (!p) {
		printf("\n");
		return;
	}
	size_t n = p->n;
	if (n < 1) {
		printf("\n");
		return;
	}
	size_t i;
	for (i = 0; i < n - 1; i++) {
		printf("%d, ", p->a[i]);
	}
	printf("%d\n", p->a[i]);
}
