#ifndef BLAS_H
#define BLAS_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

	void add_biases(float* output, float* biases, size_t F, size_t S, size_t batch_size);
	void get_bias_grads(float* bias_grads, float* grads, size_t F, size_t S, size_t batch_size);
	void fill_array(float* arr, size_t size, float val);
	void fill_array_increment(float* arr, size_t size, float start_val, float increment);
	void fill_array_rand_float(float* arr, size_t size, double mean, double sdev);
	void zero_array(float* arr, size_t size);
	float sum_array(float* arr, size_t size);
	void scale_array(float* arr, size_t size, float scalar);

#ifdef __cplusplus
}
#endif
#endif