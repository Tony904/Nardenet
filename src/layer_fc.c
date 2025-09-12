#include "layer_fc.h"
#include <omp.h>
#include "blas.h"
#include "batchnorm.h"
#include "xcuda.h"


void forward_fc(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	size_t out_n = l->out_n;
	float* Z = l->Z;
	zero_array(Z, out_n * batch_size);
	float* weights = l->weights;
	float* w0 = weights;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t n = 0; n < out_n; n++) {
			size_t offset = b * out_n + n;
			for (size_t a = 0; a < l->in_ids.n; a++) {
				size_t inl_out_n = l->in_layers[a]->out_n;
				float* inl_output = &l->in_layers[a]->output[inl_out_n * b];
				float sum = 0.0F;
				size_t i;
#pragma omp parallel for reduction(+:sum)
				for (i = 0; i < inl_out_n; i++) {
					sum += inl_output[i] * weights[i];
				}
				Z[offset] += sum;
				weights += inl_out_n;
			}
		}
		weights = w0;
	}
	if (l->batchnorm) {
		forward_batchnorm(l, batch_size);
		l->activate(l->act_inputs, l->output, out_n, batch_size);
	}
	else {
		// note: l->Z = l->act_inputs when batchnorm disabled
		add_biases(l->Z, l->biases, out_n, 1, batch_size);
		l->activate(l->Z, l->output, out_n, batch_size);
	}
	if (net->training) zero_array(l->grads, out_n * batch_size);
}

void backward_fc(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->grads;
	size_t out_n = l->out_n;
	get_activation_grads(l, batch_size);
	get_bias_grads(l->bias_grads, grads, out_n, 1, batch_size);  // note: biases = betas for batch norm
	if (l->batchnorm) backward_batchnorm(l, batch_size);
	float* weight_grads = l->weight_grads;
	float* wg0 = weight_grads;
	float* weights = l->weights;
	float* w0 = weights;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t n = 0; n < out_n; n++) {
			float grad = grads[b * out_n + n];
			for (size_t a = 0; a < l->in_ids.n; a++) {
				size_t inl_out_n = l->in_layers[a]->out_n;
				float* inl_grads = &l->in_layers[a]->grads[inl_out_n * b];
				float* inl_output = &l->in_layers[a]->output[inl_out_n * b];
				size_t i;
#pragma omp parallel for firstprivate(grad)
				for (i = 0; i < inl_out_n; i++) {
					weight_grads[i] += inl_output[i] * grad;
					inl_grads[i] += weights[i] * grad;
				}
				weight_grads += inl_out_n;
				weights += inl_out_n;
			}
		}
		weight_grads = wg0;
		weights = w0;
	}
}

void forward_fc_gpu(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	size_t out_n = l->out_n;
	float* Z = l->gpu.Z;
	zero_array_gpu(Z, out_n * batch_size);
	float* weights = l->gpu.weights;
	float* w0 = weights;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t n = 0; n < out_n; n++) {
			size_t offset = b * out_n + n;
			for (size_t a = 0; a < l->in_ids.n; a++) {
				size_t inl_out_n = l->in_layers[a]->out_n;
				float* inl_output = &l->in_layers[a]->gpu.output[inl_out_n * b];
				dot_product_gpu(weights, inl_output, inl_out_n, &Z[offset]);
				weights += inl_out_n;
			}
		}
		weights = w0;
	}
	if (l->batchnorm) {
		forward_batchnorm_gpu(l->gpu.gammas, l->gpu.biases, l->gpu.means, l->gpu.variances, l->gpu.rolling_means, l->gpu.rolling_variances, l->gpu.Z, l->gpu.Z_norm, l->gpu.act_inputs, (int)(l->w * l->h), (int)out_n, (int)batch_size);
		l->activate(l->gpu.act_inputs, l->gpu.output, out_n, batch_size);
	}
	else {
		// note: l->Z = l->act_inputs when batchnorm disabled
		add_biases_gpu(l->gpu.Z, l->gpu.biases, out_n, 1, batch_size);
		l->activate(l->gpu.Z, l->gpu.output, out_n, batch_size);
	}
	if (net->training) zero_array_gpu(l->gpu.grads, out_n * batch_size);
}

void backward_fc_gpu(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->gpu.grads;
	size_t out_n = l->out_n;
	get_activation_grads_gpu(l, batch_size);
	get_bias_grads_gpu(l->gpu.bias_grads, grads, out_n, 1, batch_size);  // note: biases = betas for batch norm
	if (l->batchnorm) backward_batchnorm_gpu(grads, l->gpu.Z, l->gpu.Z_norm, l->gpu.means, l->gpu.variances, l->gpu.gammas, l->gpu.gamma_grads, 1, (int)out_n, (int)batch_size);
	float* weight_grads = l->gpu.weight_grads;
	float* wg0 = weight_grads;
	float* weights = l->gpu.weights;
	float* w0 = weights;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t n = 0; n < out_n; n++) {
			size_t offset = b * out_n + n;
			for (size_t a = 0; a < l->in_ids.n; a++) {
				size_t inl_out_n = l->in_layers[a]->out_n;
				float* inl_grads = &l->in_layers[a]->gpu.grads[inl_out_n * b];
				float* inl_output = &l->in_layers[a]->gpu.output[inl_out_n * b];
				scale_add_array_gpu(inl_output, weight_grads, &grads[offset], inl_out_n);
				scale_add_array_gpu(weights, inl_grads, &grads[offset], inl_out_n);
				weight_grads += inl_out_n;
				weights += inl_out_n;
			}
		}
		weight_grads = wg0;
		weights = w0;
	}
}