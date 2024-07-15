#ifndef DERIVATIVES_H
#define DERIVATIVES_H


#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif

	void get_grads_sigmoid(float* grads, float* act_outputs, size_t out_n, size_t batch_size);  // derivative of sigmoid depends on it's output
	void get_grads_mish(float* grads, float* act_inputs, size_t out_n, size_t batch_size);
	void get_grads_relu(float* grads, float* act_inputs, size_t out_n, size_t batch_size);
	void get_grads_leaky_relu(float* grads, float* act_inputs, size_t out_n, size_t batch_size);
	void get_grads_tanh(float* grads, float* act_inputs, size_t out_n, size_t batch_size);
	void regularize_l1(float* weight_grads, float* weights, size_t size, float decay);
	void regularize_l2(float* weight_grads, float* weights, size_t size, float decay);
	void regularize_none(float* weight_grads, float* weights, size_t size, float decay);

#ifdef __cplusplus
}
#endif
#endif