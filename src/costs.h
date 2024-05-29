#ifndef COSTS_H
#define COSTS_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

	void get_cost_mse(float* grads, float* errors, float* output, float* truth, size_t size);

#ifdef __cplusplus
}
#endif
#endif