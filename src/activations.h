#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H


#include <math.h>


#define MISH_THRESH 20.0F


#ifdef __cplusplus
extern "C" {
#endif

	extern inline float sigmoid_x(float x) { return 1.0F / (1.0F + expf(-x)); }
	extern inline float tanh_x(float x) { return (2.0F / (1.0F + expf(-2.0F * x)) - 1.0F); }
	extern inline float relu_x(float x) { return x * (x > 0.0F); }
	extern inline float leaky_x(float x) { return x > 0.0F ? x : 0.1F * x; }
	extern inline float softplus_x(float x, float t) { return (x > t) ? x : (x < -t) ? expf(x) : logf(expf(x) + 1.0F); }
	extern inline float mish_x(float x, float thresh) { return x * tanh_x(softplus_x(x, thresh)); }
	void activate_none(float* Z, float* output, size_t size, size_t batch_size);
	void activate_mish(float* Z, float* output, size_t size, size_t batch_size);
	void activate_relu(float* Z, float* output, size_t size, size_t batch_size);
	void activate_sigmoid(float* Z, float* output, size_t size, size_t batch_size);
	void activate_leaky_relu(float* Z, float* output, size_t size, size_t batch_size);
	void activate_softmax(float* Z, float* output, size_t size, size_t batch_size);

#ifdef __cplusplus
}
#endif
#endif