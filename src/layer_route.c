#include "layer_conv.h"
#include "utils.h"
#include "derivatives.h"
#include "blas.h"
#include "xcuda.h"


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

void forward_route_gpu(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* Z = l->Z;
	float* Z0 = Z;
	for (size_t a = 0; a < l->in_ids.n; a++) {
		float* inl_output = l->in_layers[a]->output;
		size_t inl_out_n = l->in_layers[a]->out_n * batch_size;
		copy_array_gpu(inl_output, Z, inl_out_n);
		Z += inl_out_n;
	}
	if (l->activation) l->activate(Z0, l->output, l->out_n, net->batch_size);
	if (net->training) zero_array_gpu(l->grads, l->out_n * batch_size);
}

void backward_route(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->grads;
	if (l->activation) get_activation_grads(l, batch_size);
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

void backward_route_gpu(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->grads;
	if (l->activation) get_activation_grads_gpu(l, batch_size);
	for (size_t a = 0; a < l->in_ids.n; a++) {
		float* inl_grads = l->in_layers[a]->grads;
		size_t inl_out_n = l->in_layers[a]->out_n * batch_size;
		copy_array_gpu(grads, inl_grads, inl_out_n);
		grads += inl_out_n;
	}
}