#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H


#include <math.h>


#ifdef __cplusplus
extern "C" {
#endif


	extern inline float tanh_x(float x) { return (2.f / (1.f + expf(-2.f * x)) - 1.f); }
	extern inline float relu_x(float x) { return x * (x > 0); }
	extern inline float softplus_x(float x, float t) { return (x > t) ? x : (x < -t) ? expf(x) : logf(expf(x) + 1); }
	extern inline float mish_x(float x, float thresh) { return x * tanh_x(softplus_x(x, thresh)); }


#ifdef __cplusplus
}
#endif
#endif