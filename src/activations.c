#include "activations.h"



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

void activate_tanh(float* Z, float* output, size_t out_n, size_t batch_size) {
	size_t n = out_n * batch_size;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		output[i] = tanh_x(Z[i]);
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
	size_t b;
#pragma omp parallel for firstprivate(out_n)
	for (b = 0; b < batch_size; b++) {
		float* z = &Z[b * out_n];
		float* a = &output[b * out_n];
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

#pragma warning (suppress:4100)
void activate_none(float* Z, float* output, size_t out_n, size_t batch_size) {
}