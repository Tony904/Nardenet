#include "layer_conv.h"
#include "utils.h"
#include "derivatives.h"
#include "blas.h"
#include "xcuda.h"


void forward_route(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* Z = l->Z;
	float* Z0 = Z;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t a = 0; a < l->in_ids.n; a++) {
			size_t inl_out_n = l->in_layers[a]->out_n;
			float* inl_output = &l->in_layers[a]->output[inl_out_n * b];
			size_t i;
#pragma omp parallel for
			for (i = 0; i < inl_out_n; i++) {
				Z[i] = inl_output[i];
			}
			Z += inl_out_n;
		}
	}
	if (l->activation) l->activate(Z0, l->output, l->out_n, batch_size);
	if (net->training) zero_array(l->grads, l->out_n * batch_size);
}

void backward_route(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->grads;
	if (l->activation) get_activation_grads(l, batch_size);
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t a = 0; a < l->in_ids.n; a++) {
			size_t inl_out_n = l->in_layers[a]->out_n;
			float* inl_grads = &l->in_layers[a]->grads[inl_out_n * b];
			size_t i;
#pragma omp parallel for
			for (i = 0; i < inl_out_n; i++) {
				inl_grads[i] += grads[i];
			}
			grads += inl_out_n;
		}
	}
}

#ifdef GPU
void forward_route_gpu(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* Z = l->gpu.Z;
	float* Z0 = Z;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t a = 0; a < l->in_ids.n; a++) {
			size_t inl_out_n = l->in_layers[a]->out_n;
			float* inl_output = &l->in_layers[a]->gpu.output[inl_out_n * b];
			copy_array_gpu(inl_output, Z, (int)inl_out_n);
			Z += inl_out_n;
		}
	}
	
	if (l->activation) l->activate(Z0, l->gpu.output, l->out_n, batch_size);
	if (net->training) zero_array_gpu(l->gpu.grads, (int)(l->out_n * batch_size));
}

void backward_route_gpu(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->gpu.grads;
	if (l->activation) get_activation_grads_gpu(l, (int)batch_size);
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t a = 0; a < l->in_ids.n; a++) {
			size_t inl_out_n = l->in_layers[a]->out_n;
			float* inl_grads = &l->in_layers[a]->gpu.grads[inl_out_n * b];
			add_arrays_gpu(grads, inl_grads, (int)inl_out_n);
			grads += inl_out_n;
		}
	}
}
#else
#pragma warning (suppress:4100)
void forward_route_gpu(layer* l, network* net) {
	gpu_not_defined();
}
#pragma warning (suppress:4100)
void backward_route_gpu(layer* l, network* net) {
	gpu_not_defined();
}
#endif