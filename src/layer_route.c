#include "layer_conv.h"
#include "utils.h"
#include "derivatives.h"


void forward_route(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* Z = l->Z;
	for (size_t a = 0; a < l->in_ids.n; a++) {
		float* inl_output = l->in_layers[a]->output;
		size_t inl_out_n = l->in_layers[a]->out_n * batch_size;
		size_t i;
#pragma omp parallel for
		for (i = 0; i < inl_out_n; i++) {
			Z[i] = inl_output[i];
		}
		Z += inl_out_n;
	}
	Z = l->Z;
	if (l->activation) l->activate(Z, l->output, l->out_n, net->batch_size);
	if (net->training) zero_array(l->grads, l->out_n * batch_size);
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
	for (size_t a = 0; a < l->in_ids.n; a++) {
		float* inl_grads = l->in_layers[a]->grads;
		size_t inl_out_n = l->in_layers[a]->out_n * batch_size;
		size_t i;
#pragma omp parallel for
		for (i = 0; i < inl_out_n; i++) {
			inl_grads[i] = grads[i];
		}
		grads += inl_out_n;
	}
}