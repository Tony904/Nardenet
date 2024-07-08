#include "batchnorm.h"
#include <math.h>


void batch_normalize(layer* l, size_t batch_size) {
	float* Z = l->Z;
	float* Z_norm = l->Z_norm;
	float* act_inputs = l->act_inputs;
	float* means = l->means;
	float* variances = l->variances;
	float* gammas = l->gammas;
	float* betas = l->biases;
	float* rolling_means = l->rolling_means;
	float* rolling_variances = l->rolling_variances;
	size_t F = l->n_filters;
	size_t S = l->out_w * l->out_h;
	size_t B = batch_size;
	size_t out_n = l->out_n;
	float SB = (float)(S * B);
	// calculate means
	size_t f;
#pragma omp parallel for firstprivate(S, B, out_n, SB)
	for (f = 0; f < F; f++) {
		float sum = 0.0F;
		size_t fS = f * S;
		for (size_t b = 0; b < B; b++) {
			size_t offset = b * out_n + fS;
			for (size_t s = 0; s < S; s++) {
				sum += Z[offset + s];
			}
		}
		means[f] = sum / SB;
	}
	// calculate variances
#pragma omp parallel for firstprivate(S, B, out_n, SB)
	for (f = 0; f < F; f++) {
		float sum = 0.0F;
		size_t fS = f * S;
		for (size_t b = 0; b < B; b++) {
			size_t offset = b * out_n + fS;
			for (size_t s = 0; s < S; s++) {
				float dev = Z[offset + s] - means[f];
				sum += dev * dev;
			}
		}
		variances[f] = sum / SB;
	}
	// calculate Z_norm
#pragma omp parallel for firstprivate(S, B, out_n)
	for (f = 0; f < F; f++) {
		float mean = means[f];
		float variance = variances[f];
		float sddev = sqrtf(variance + 0.00001F);
		rolling_means[f] = (mean * 0.1F) + (rolling_means[f] * 0.9F);
		rolling_variances[f] = (variance * 0.1F) + (rolling_variances[f] * 0.9F);
		size_t fS = f * S;
		for (size_t b = 0; b < B; b++) {
			size_t offset = b * out_n + fS;
			for (size_t s = 0; s < S; s++) {
				size_t i = offset + s;
				float znorm = (Z[i] - means[f]) / sddev;
				Z_norm[i] = znorm;
				act_inputs[i] = znorm * gammas[f] + betas[f];
			}
		}
	}
}