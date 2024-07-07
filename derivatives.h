#ifndef DERIVATIVES_H
#define DERIVATIVES_H


#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif

	void get_grads_sigmoid(float* grads, float* output, size_t size);
	void get_grads_mish(float* grads, float* act_input, size_t size);
	void get_grads_relu(float* grads, float* act_input, size_t size);
	void get_grads_leaky_relu(float* grads, float* act_input, size_t size);
	void regularize_l1(float* weight_grads, float* weights, size_t size, float decay);
	void regularize_l2(float* weight_grads, float* weights, size_t size, float decay);
	void regularize_none(float* weight_grads, float* weights, size_t size, float decay);

#ifdef __cplusplus
}
#endif
#endif