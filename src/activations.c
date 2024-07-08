#include "activations.h"


void activate_relu(float* Z, float* output, size_t out_n, size_t batch_size) {
	size_t i;
#pragma omp parallel for
	for (i = 0; i < out_n; i++) {
		output[i] = relu_x(Z[i]);
	}
}

void activate_leaky_relu(float* Z, float* output, size_t out_n, size_t batch_size) {
	size_t i;
#pragma omp parallel for
	for (i = 0; i < out_n; i++) {
		output[i] = leaky_x(Z[i]);
	}
}

void activate_mish(float* Z, float* output, size_t out_n, size_t batch_size) {
	size_t i;
#pragma omp parallel for
	for (i = 0; i < out_n; i++) {
		output[i] = mish_x(Z[i], MISH_THRESH);
	}
}

void activate_sigmoid(float* Z, float* output, size_t out_n, size_t batch_size) {
	size_t i;
#pragma omp parallel for
	for (i = 0; i < out_n; i++) {
		output[i] = sigmoid_x(Z[i]);
	}
}

void activate_softmax(float* Z, float* output, size_t out_n, size_t batch_size) {
	float* dst = output;
	float* src = Z;
	float sum = 0.0F;
	// Calculate maxval to then subtract for numerical stability
	// https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
	float maxval = src[0];
	size_t i;
	for (i = 1; i < out_n; i++) {
		if (src[i] > maxval) {
			maxval = src[i];
		}
	}
	//#pragma omp parallel for
	for (i = 0; i < out_n; i++) {
		float e = expf(src[i] - maxval);
		sum += e;
		dst[i] = e;
	}
	for (i = 0; i < out_n; i++) dst[i] /= sum;
}

#pragma warning(suppress:4100)  // unreferenced formal parameter
void activate_none(float* Z, float* output, size_t out_n) {
	out_n;
}