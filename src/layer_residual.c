#include "layer_residual.h"
#include "derivatives.h"
#include "utils.h"


void forward_residual(layer* l, network* net) {
	size_t n = l->out_n * l->batch_size;
	float* l_Z = l->Z;
	float* inl0_output = l->in_layers[0]->output;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		l_Z[i] = inl0_output[i];
	}
	for (size_t a = 1; a < l->in_ids.n; a++) {
		float* inl_output = l->in_layers[a]->output;
#pragma omp parallel for
		for (i = 0; i < n; i++) {
			l_Z[i] += inl_output[i];
		}
	}
	if (l->activation) l->activate(l_Z, l->output, l->out_n, l->batch_size);
	else l->output = l_Z;
	if (net->training) zero_array(l->grads, n);
}

void backward_residual(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->grads;
	if (l->activation) {
		if (l->activation == ACT_MISH) get_grads_mish(grads, l->Z, l->out_n, batch_size);  // dC/da * da/dz
		else if (l->activation == ACT_RELU) get_grads_relu(grads, l->Z, l->out_n, batch_size);
		else if (l->activation == ACT_LEAKY) get_grads_leaky_relu(grads, l->Z, l->out_n, batch_size);
		else if (l->activation == ACT_SIGMOID) get_grads_sigmoid(grads, l->Z, l->out_n, batch_size);
		else if (l->activation == ACT_TANH) get_grads_tanh(grads, l->Z, l->out_n, batch_size);
		else {
			printf("Incorrect or unsupported activation function.\n");
			exit(EXIT_FAILURE);
		}
	}
	size_t n = batch_size * l->out_n;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t a = 0; a < l->in_ids.n; a++) {
			float* inl_grads = l->in_layers[a]->grads;
			size_t i;
#pragma omp parallel for
			for (i = 0; i < n; i++) {
				inl_grads[i] += grads[i];
			}
		}
	}
}