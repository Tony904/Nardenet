#include "activations.h"

#pragma warning(disable:4100)  // temporary
void activate_relu(float* Z, float* output, size_t out_n, size_t batch_size) {
	size_t n = out_n * batch_size;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		output[i] = relu_x(Z[i]);
	}
}

void activate_leaky_relu(float* Z, float* output, size_t out_n, size_t batch_size) {
	size_t n = out_n * batch_size;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		output[i] = leaky_x(Z[i]);
	}
}

void activate_mish(float* Z, float* output, size_t out_n, size_t batch_size) {
	size_t n = out_n * batch_size;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		output[i] = mish_x(Z[i], MISH_THRESH);
	}
}

void activate_sigmoid(float* Z, float* output, size_t out_n, size_t batch_size) {
	size_t n = out_n * batch_size;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		output[i] = sigmoid_x(Z[i]);
	}
}

void activate_softmax(float* Z, float* output, size_t out_n, size_t batch_size) {
	// Calculate maxval to then subtract for numerical stability
	// https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
	size_t s;
#pragma omp parallel for firstprivate(out_n)
	for (s = 0; s < batch_size; s++) {
		float* z = &Z[s * out_n];
		float* a = &output[s * out_n];
		float maxval = z[0];
		for (size_t i = 1; i < out_n; i++) {
			if (z[i] > maxval) maxval = z[i];
		}
		float sum = 0.0F;
		for (size_t i = 0; i < out_n; i++) {
			float e = expf(z[i] - maxval);
			sum += e;
			a[i] = e;
		}
		for (size_t i = 0; i < out_n; i++) a[i] /= sum;
	}
}

#pragma warning(suppress:4100)  // unreferenced formal parameter
void activate_none(float* Z, float* output, size_t out_n, size_t batch_size) {
	out_n;
}