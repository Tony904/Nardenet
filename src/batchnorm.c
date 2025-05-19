#include "batchnorm.h"
#include <math.h>
#include "utils.h"


#define MAX_FILTERS 10240


void forward_batchnorm(layer* l, size_t batch_size) {
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
	// calculate Z_norm and apply scaling and shift
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

void backward_batchnorm(layer* l, size_t batch_size) {
	float* grads = l->grads;
	float* Z = l->Z;
	float* Z_norm = l->Z_norm;
	float* means = l->means;
	float* variances = l->variances;
	float* gammas = l->gammas;
	float* gamma_grads = l->gamma_grads;
	size_t out_n = l->out_n;

	size_t F = l->n_filters;
	size_t S = l->out_w * l->out_h;
	size_t B = batch_size;

	// Calculate gradients wrt gamma, which is dL/da * dznorm/dgamma (dznorm/dgamma is local grad of gamma which is just Z_norm)
	size_t f;
#pragma omp parallel for firstprivate(out_n)
	for (f = 0; f < F; f++) {
		float sum = 0.0F;
		size_t fS = f * S;
		for (size_t b = 0; b < B; b++) {
			size_t offset = b * out_n + fS;
			for (size_t s = 0; s < S; s++) {
				sum += grads[offset + s] * Z_norm[offset + s];
			}
		}
		gamma_grads[f] = sum;  // dL/dgammas
	}

	// Get d(act_inputs)/dZnorm
	zero_array(Z_norm, out_n * batch_size);
#pragma omp parallel for firstprivate(out_n)
	for (f = 0; f < F; f++) {
		size_t fS = f * S;
		for (size_t b = 0; b < B; b++) {
			size_t offset = b * out_n + fS;
			for (size_t s = 0; s < S; s++) {
				grads[offset + s] *= gammas[f];
			}
		}
	}
	// grads is now dL/dznorm

	// Now we need to get the gradients wrt means and variances and then use those to get the gradients wrt Z.
	float mean_grads[MAX_FILTERS] = { 0 };/*
	if (F > MAX_FILTERS) {
		printf("Layer %d has too many filters. Max filters = %d\n", l->id, MAX_FILTERS);
		wait_for_key_then_exit();
	}*/
#pragma omp parallel for firstprivate(out_n)
	for (f = 0; f < F; f++) {
		float sum = 0.0F;
		size_t fS = f * S;
		for (size_t b = 0; b < B; b++) {
			size_t offset = b * out_n + fS;
			for (size_t s = 0; s < S; s++) {
				sum += grads[offset + s];
			}
		}
		mean_grads[f] = sum * (-1.0F/sqrt(variances[f] + 0.00001F));
	}

	float variance_grads[MAX_FILTERS] = { 0 };
#pragma omp parallel for firstprivate(out_n)
	for (f = 0; f < F; f++) {
		float sum = 0.0F;
		size_t fS = f * S;
		for (size_t b = 0; b < B; b++) {
			size_t offset = b * out_n + fS;
			for (size_t s = 0; s < S; s++) {
				size_t i = offset + s;
				sum += grads[i] * (Z[i] - means[f]);
			}
		}
		variance_grads[f] *= -0.5F * pow(variances[f] + 0.00001F, (float)(-3.0F / 2.0F));
	}

#pragma omp parallel for firstprivate(out_n)
	for (f = 0; f < F; f++) {
		size_t fS = f * S;
		float SB = (float)(S * B);
		for (size_t b = 0; b < B; b++) {
			size_t offset = b * out_n + fS;
			for (size_t s = 0; s < S; s++) {
				size_t i = offset + s;
				grads[i] = grads[i] * 1.0F / (sqrt(variances[f]) + 0.00001F) + variance_grads[f] * 2.0F * (grads[i] - means[f]) / SB + mean_grads[f] /SB;
			}
		}
	}
}