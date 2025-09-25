#include "layer_fc.h"
#include <omp.h>
#include <string.h>
#include "blas.h"
#include "batchnorm.h"
#include "xcuda.h"

#define BUFFSIZE 256

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


void forward_fc_cpu_gpu_compare(layer* l, network* net) {
	char buff[BUFFSIZE] = { 0 };

	size_t batch_size = net->batch_size;
	size_t out_n = l->out_n;
	float* Z_cpu = l->Z;
	float* Z_gpu = l->gpu.Z;
	zero_array(Z_cpu, out_n * batch_size);
	zero_array_gpu(Z_gpu, (int)(out_n * batch_size));
	compare_cpu_gpu_arrays(l->Z, l->gpu.Z, out_n * batch_size, l->id, "forward fc, Z, after zeroing");

	float* weights_cpu = l->weights;
	float* weights_gpu = l->gpu.weights;

	compare_cpu_gpu_arrays(l->weights, l->gpu.weights, l->n_weights, l->id, "forward fc, weights before dot products");

	for (size_t b = 0; b < batch_size; b++) {
		for (size_t n = 0; n < out_n; n++) {
			size_t offset = b * out_n + n;
			for (size_t a = 0; a < l->in_ids.n; a++) {
				size_t inl_out_n = l->in_layers[a]->out_n;
				float* inl_output_cpu = &l->in_layers[a]->output[inl_out_n * b];
				float* inl_output_gpu = &l->in_layers[a]->gpu.output[inl_out_n * b];

				memset(buff, 0, sizeof(buff));
				snprintf(buff, BUFFSIZE, "%s%zu", "forward fc, inl_output, pre-dot product, b=", b);
				compare_cpu_gpu_arrays(inl_output_cpu, inl_output_gpu, inl_out_n, l->id, buff);

				float sum = 0.0F;
				size_t i;
#pragma omp parallel for reduction(+:sum)
				for (i = 0; i < inl_out_n; i++) {
					sum += inl_output_cpu[i] * weights_cpu[i];
				}
				Z_cpu[offset] += sum;

				dot_product_gpu(weights_gpu, inl_output_gpu, (int)inl_out_n, &Z_gpu[offset]);

				memset(buff, 0, sizeof(buff));
				snprintf(buff, BUFFSIZE, "%s, b=%zu, n=%zu", "forward fc, Z (dot product result), post-dot product", b, n);
				compare_cpu_gpu_arrays(Z_cpu, Z_gpu, 1, l->id, buff);

				weights_cpu += inl_out_n;
				weights_gpu += inl_out_n;
			}
		}
		weights_cpu = l->weights;
		weights_gpu = l->gpu.weights;
	}
	if (l->batchnorm) {

		compare_cpu_gpu_arrays(l->gammas, l->gpu.gammas, l->n_filters, l->id, "forward conv, gammas, pre-batchnorm");
		compare_cpu_gpu_arrays(l->biases, l->gpu.biases, l->n_filters, l->id, "forward conv, biases, pre-batchnorm");
		compare_cpu_gpu_arrays(l->means, l->gpu.means, l->n_filters, l->id, "forward conv, means, pre-batchnorm");
		compare_cpu_gpu_arrays(l->variances, l->gpu.variances, l->n_filters, l->id, "forward conv, variances, pre-batchnorm");
		compare_cpu_gpu_arrays(l->rolling_means, l->gpu.rolling_means, l->n_filters, l->id, "forward conv, rolling means, pre-batchnorm");
		compare_cpu_gpu_arrays(l->rolling_variances, l->gpu.rolling_variances, l->n_filters, l->id, "forward conv, rolling variances, pre-batchnorm");
		compare_cpu_gpu_arrays(l->Z, l->gpu.Z, l->n_filters, l->id, "forward conv, Z, pre-batchnorm");
		compare_cpu_gpu_arrays(l->Z_norm, l->gpu.Z_norm, l->n_filters, l->id, "forward conv, Z_norm, pre-batchnorm");
		compare_cpu_gpu_arrays(l->act_inputs, l->gpu.act_inputs, l->n_filters, l->id, "forward, act_inputs, pre-batchnorm");

		forward_batchnorm(l, batch_size);
		forward_batchnorm_gpu(l->gpu.gammas, l->gpu.biases, l->gpu.means, l->gpu.variances, l->gpu.rolling_means, l->gpu.rolling_variances, l->gpu.Z, l->gpu.Z_norm, l->gpu.act_inputs, (int)(l->w * l->h), (int)out_n, (int)batch_size);
		
		compare_cpu_gpu_arrays(l->gammas, l->gpu.gammas, l->n_filters, l->id, "forward conv, gammas, post-batchnorm");
		compare_cpu_gpu_arrays(l->biases, l->gpu.biases, l->n_filters, l->id, "forward conv, biases, post-batchnorm");
		compare_cpu_gpu_arrays(l->means, l->gpu.means, l->n_filters, l->id, "forward conv, means, post-batchnorm");
		compare_cpu_gpu_arrays(l->variances, l->gpu.variances, l->n_filters, l->id, "forward conv, variances, post-batchnorm");
		compare_cpu_gpu_arrays(l->rolling_means, l->gpu.rolling_means, l->n_filters, l->id, "forward conv, rolling means, post-batchnorm");
		compare_cpu_gpu_arrays(l->rolling_variances, l->gpu.rolling_variances, l->n_filters, l->id, "forward conv, rolling variances, post-batchnorm");
		compare_cpu_gpu_arrays(l->Z, l->gpu.Z, l->n_filters, l->id, "forward conv, Z, post-batchnorm");
		compare_cpu_gpu_arrays(l->Z_norm, l->gpu.Z_norm, l->n_filters, l->id, "forward conv, Z_norm, post-batchnorm");
		compare_cpu_gpu_arrays(l->act_inputs, l->gpu.act_inputs, l->n_filters, l->id, "forward, act_inputs, post-batchnorm");
		
		activate(l->act_inputs, l->output, out_n, batch_size, l->activation, 0);
		activate(l->gpu.act_inputs, l->gpu.output, out_n, batch_size, l->activation, 1);

		compare_cpu_gpu_arrays(l->output, l->gpu.output, out_n * batch_size, l->id, "forward conv, output, post-activation");
	}
	else {
		// note: l->Z = l->act_inputs when batchnorm disabled
		add_biases(l->Z, l->biases, out_n, 1, batch_size);
		add_biases_gpu(l->gpu.Z, l->gpu.biases, (int)out_n, 1, (int)batch_size);

		compare_cpu_gpu_arrays(l->Z, l->gpu.Z, out_n, l->id, "forward fc, Z after adding biases");

		activate(l->Z, l->output, out_n, batch_size, l->activation, 0);
		activate(l->gpu.Z, l->gpu.output, out_n, batch_size, l->activation, 1);

		compare_cpu_gpu_arrays(l->output, l->gpu.output, out_n * batch_size, l->id, "forward fc, output after activation");
	}
	if (net->training) {
		zero_array(l->grads, out_n * batch_size);
		zero_array_gpu(l->gpu.grads, (int)(out_n * batch_size));

		compare_cpu_gpu_arrays(l->grads, l->gpu.grads, out_n * batch_size, l->id, "forward fc, grads after zeroing");
	}
}

void backward_fc_cpu_gpu_compare(layer* l, network* net) {
	char buff[BUFFSIZE] = { 0 };

	size_t batch_size = net->batch_size;
	float* grads_cpu = l->grads;
	float* grads_gpu = l->gpu.grads;
	size_t out_n = l->out_n;

	compare_cpu_gpu_arrays(grads_cpu, grads_gpu, batch_size * l->out_n, l->id, "backward fc, grads, pre-activation grads");

	get_activation_grads(l, batch_size);
	get_activation_grads_gpu(l, batch_size);

	compare_cpu_gpu_arrays(grads_cpu, grads_gpu, batch_size * l->out_n, l->id, "backward fc, grads, post-activation grads");

	get_bias_grads(l->bias_grads, grads_cpu, out_n, 1, batch_size);  // note: biases = betas for batch norm
	get_bias_grads_gpu(l->gpu.bias_grads, grads_gpu, (int)out_n, 1, (int)batch_size);

	compare_cpu_gpu_arrays(l->bias_grads, l->gpu.bias_grads, out_n, l->id, "backward fc, bias grads");

	if (l->batchnorm) {
		backward_batchnorm(l, batch_size);
		backward_batchnorm_gpu(grads_gpu, l->gpu.Z, l->gpu.Z_norm, l->gpu.means, l->gpu.variances, l->gpu.gammas, l->gpu.gamma_grads, 1, (int)out_n, (int)batch_size);
	}
	float* weight_grads_cpu = l->weight_grads;
	float* weights_cpu = l->weights;
	float* weight_grads_gpu = l->gpu.weight_grads;
	float* weights_gpu = l->gpu.weights;

	compare_cpu_gpu_arrays(l->weights, l->gpu.weights, l->n_weights, l->id, "backward fc, weights");
	compare_cpu_gpu_arrays(l->weight_grads, l->gpu.weight_grads, l->n_weights, l->id, "backward fc, weight grads");

	for (size_t b = 0; b < batch_size; b++) {
		for (size_t n = 0; n < out_n; n++) {
			float grad_cpu = grads_cpu[b * out_n + n];
			float* grad_gpu = &grads_gpu[b * out_n + n];
			for (size_t a = 0; a < l->in_ids.n; a++) {
				size_t inl_out_n = l->in_layers[a]->out_n;
				float* inl_grads_cpu = &l->in_layers[a]->grads[inl_out_n * b];
				float* inl_grads_gpu = &l->in_layers[a]->gpu.grads[inl_out_n * b];

				memset(buff, 0, sizeof(buff));
				snprintf(buff, BUFFSIZE, "%s, b=%zu, n=%zu", "backward fc, inl_grads, pre-math", b, n);
				compare_cpu_gpu_arrays(inl_grads_cpu, inl_grads_gpu, inl_out_n, l->id, buff);

				float* inl_output_cpu = &l->in_layers[a]->output[inl_out_n * b];
				float* inl_output_gpu = &l->in_layers[a]->gpu.output[inl_out_n * b];

				memset(buff, 0, sizeof(buff));
				snprintf(buff, BUFFSIZE, "%s, b=%zu, n=%zu", "backward fc, inl_outputs, pre-math", b, n);
				compare_cpu_gpu_arrays(inl_output_cpu, inl_output_gpu, inl_out_n, l->id, buff);

				memset(buff, 0, sizeof(buff));
				snprintf(buff, BUFFSIZE, "%s, b=%zu, n=%zu", "backward fc, weight grads, pre-math", b, n);
				compare_cpu_gpu_arrays(weight_grads_cpu, weight_grads_gpu, inl_out_n, l->id, buff);

				size_t i;
#pragma omp parallel for firstprivate(grad_cpu)
				for (i = 0; i < inl_out_n; i++) {
					weight_grads_cpu[i] += inl_output_cpu[i] * grad_cpu;
					inl_grads_cpu[i] += weights_cpu[i] * grad_cpu;
				}

				scale_add_array_gpu(inl_output_gpu, weight_grads_gpu, grad_gpu, (int)inl_out_n);
				scale_add_array_gpu(weights_gpu, inl_grads_gpu, grad_gpu, (int)inl_out_n);

				memset(buff, 0, sizeof(buff));
				snprintf(buff, BUFFSIZE, "%s, b=%zu, n=%zu", "backward fc, weight grads, post-math", b, n);
				compare_cpu_gpu_arrays(weight_grads_cpu, weight_grads_gpu, inl_out_n, l->id, buff);

				memset(buff, 0, sizeof(buff));
				snprintf(buff, BUFFSIZE, "%s, b=%zu, n=%zu", "backward fc, inl_grads, post-math", b, n);
				compare_cpu_gpu_arrays(inl_grads_cpu, inl_grads_gpu, inl_out_n, l->id, buff);

				weight_grads_cpu += inl_out_n;
				weights_cpu += inl_out_n;

				weight_grads_gpu += inl_out_n;
				weights_gpu += inl_out_n;
			}
		}
		weight_grads_cpu = l->weight_grads;
		weights_cpu = l->weights;
		weight_grads_gpu = l->gpu.weight_grads;
		weights_gpu = l->gpu.weights;
	}
}


void forward_fc_gpu(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	size_t out_n = l->out_n;
	float* Z = l->gpu.Z;
	zero_array_gpu(Z, (int)(out_n * batch_size));
	float* weights = l->gpu.weights;
	float* w0 = weights;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t n = 0; n < out_n; n++) {
			size_t offset = b * out_n + n;
			for (size_t a = 0; a < l->in_ids.n; a++) {
				size_t inl_out_n = l->in_layers[a]->out_n;
				float* inl_output = &l->in_layers[a]->gpu.output[inl_out_n * b];
				dot_product_gpu(weights, inl_output, (int)inl_out_n, &Z[offset]);
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
		add_biases_gpu(l->gpu.Z, l->gpu.biases, (int)out_n, 1, (int)batch_size);
		l->activate(l->gpu.Z, l->gpu.output, out_n, batch_size);
	}
	if (net->training) zero_array_gpu(l->gpu.grads, (int)(out_n * batch_size));
}

void backward_fc_gpu(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->gpu.grads;
	size_t out_n = l->out_n;
	get_activation_grads_gpu(l, batch_size);
	get_bias_grads_gpu(l->gpu.bias_grads, grads, (int)out_n, 1, (int)batch_size);  // note: biases = betas for batch norm
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
				scale_add_array_gpu(inl_output, weight_grads, &grads[offset], (int)inl_out_n);
				scale_add_array_gpu(weights, inl_grads, &grads[offset], (int)inl_out_n);
				weight_grads += inl_out_n;
				weights += inl_out_n;
			}
		}
		weight_grads = wg0;
		weights = w0;
	}
}