#include "costs.h"


void get_cost_mse(float* grads, float* errors, float* output, float* truth, size_t size) {
#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		float delta = truth[i] - output[i];
		errors[i] = delta * delta;
		grads[i] = delta;
	}
}

//void get_cost_bce() {
//
//}
//
//void get_cost_cce() {
//
//}