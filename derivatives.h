#ifndef DERIVATIVES_H
#define DERIVATIVES_H


#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif

	extern inline float dmish_dx(float x);
	extern inline float dsigmoid_dx(float x);

	void get_grads_sigmoid(float* grads, float* output, size_t size);
	void get_grads_mish(float* grads, float* act_input, size_t size);

#ifdef __cplusplus
}
#endif
#endif