#include "blas.h"
#include "utils.h"


#pragma warning(suppress:4100) // temporary
void add_biases(float* output, float* biases, size_t n_filters, size_t spatial, size_t batch_size) {
	size_t B = (size_t)batch_size;
	size_t out_n = n_filters * spatial;
	size_t f;
#pragma omp parallel for firstprivate(out_n)
	for (f = 0; f < n_filters; f++) {
		size_t fS = f * spatial;
		for (size_t b = 0; b < B; b++) {
			size_t offset = b * out_n + fS;
			for (size_t s = 0; s < spatial; s++) {
				output[offset + s] += biases[f];
			}
		}
	}
}

void get_bias_grads(float* bias_grads, float* grads, size_t n_filters, size_t spatial, size_t batch_size) {
	size_t B = batch_size;
	size_t out_n = n_filters * spatial;
	size_t f;
#pragma omp parallel for firstprivate(out_n, spatial)
	for (f = 0; f < n_filters; f++) {
		float sum = 0.0F;
		size_t fS = f * spatial;
		for (size_t b = 0; b < B; b++) {
			size_t offset = b * out_n + fS;
			for (size_t s = 0; s < spatial; s++) {
				sum += grads[offset + s];
			}
		}
		bias_grads[f] += sum;  // += because they will be divided by batch size during update step
	}
}

void fill_array(float* arr, size_t size, float val) {
	size_t i;
#pragma omp parallel for
	for (i = 0; i < size; i++) {
		arr[i] = val;
	}
}

void zero_array(float* arr, size_t size) {
	size_t i;
#pragma omp parallel for
	for (i = 0; i < size; i++) {
		arr[i] = 0.0F;
	}
}

void fill_array_increment(float* arr, size_t size, float start_val, float increment) {
	size_t i;
#pragma omp parallel for
	for (i = 0; i < size; i++) {
		arr[i] = start_val + (float)i * increment;
	}
}

void fill_array_rand_float(float* arr, size_t size, double mean, double sdev) {
	for (size_t i = 0; i < size; i++) {
		arr[i] = (float)randn(mean, sdev);
	}
}

float sum_array(float* arr, size_t size) {
	float sum = 0.0F;
	size_t i;
#pragma omp parallel for reduction(+:sum)
	for (i = 0; i < size; i++) {
		sum += arr[i];
	}
	return sum;
}

void scale_array(float* arr, size_t size, float scalar) {
	size_t i;
#pragma omp parallel for
	for (i = 0; i < size; i++) {
		arr[i] *= scalar;
	}
}