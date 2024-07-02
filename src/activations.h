#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H


#include <math.h>
#include "network.h"


#define MISH_THRESH 20.f


#ifdef __cplusplus
extern "C" {
#endif

	extern inline float sigmoid_x(float x) { return 1.0F / (1.0F + expf(-x)); }
	extern inline float tanh_x(float x) { return (2.0F / (1.0F + expf(-2.0F * x)) - 1.0F); }
	extern inline float relu_x(float x) { return x * (x > 0.0F); }
	extern inline float leaky_x(float x) { return x > 0.0F ? x : 0.1F * x; }
	extern inline float softplus_x(float x, float t) { return (x > t) ? x : (x < -t) ? expf(x) : logf(expf(x) + 1.0F); }
	extern inline float mish_x(float x, float thresh) { return x * tanh_x(softplus_x(x, thresh)); }
	void activate_none(layer* l);
	void activate_mish(layer* l);
	void activate_relu(layer* l);
	void activate_sigmoid(layer* l);
	void activate_leaky_relu(layer* l);
	void activate_softmax(layer* l);

#ifdef __cplusplus
}
#endif
#endif