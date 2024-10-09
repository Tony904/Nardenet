#include "layer_conv.h"
#include "utils.h"
#include "derivatives.h"


void forward_route(layer* l, network* net) {
	size_t n = l->out_n * net->batch_size;
	float* Z = l->Z;
	float* inl0_output = l->in_layers[0]->output;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		Z[i] = inl0_output[i];
	}
	for (size_t a = 1; a < l->in_ids.n; a++) {
		float* inl_output = l->in_layers[a]->output;
#pragma omp parallel for
		for (i = 0; i < n; i++) {
			Z[i] += inl_output[i];
		}
	}
	if (l->activation) l->activate(Z, l->output, l->out_n, net->batch_size);
	if (net->training) zero_array(l->grads, n);
}

void backward_route(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->grads;
	if (l->activation) {
		if (l->activation == ACT_MISH) get_grads_mish(grads, l->act_inputs, l->out_n, batch_size);  // dC/da * da/dz
		else if (l->activation == ACT_RELU) get_grads_relu(grads, l->act_inputs, l->out_n, batch_size);
		else if (l->activation == ACT_LEAKY) get_grads_leaky_relu(grads, l->act_inputs, l->out_n, batch_size);
		else if (l->activation == ACT_SIGMOID) get_grads_sigmoid(grads, l->output, l->out_n, batch_size);
		else if (l->activation == ACT_SOFTMAX);
		else if (l->activation == ACT_TANH) get_grads_tanh(grads, l->act_inputs, l->out_n, batch_size);
		else {
			printf("Incorrect or unsupported activation function.\n");
			wait_for_key_then_exit();
		}
	}
	size_t N = batch_size * l->out_n;
	for (size_t a = 0; a < l->in_ids.n; a++) {
		float* inl_grads = l->in_layers[a]->grads;
		size_t i;
#pragma omp parallel for
		for (i = 0; i < N; i++) {
			inl_grads[i] += grads[i];
		}
	}
}