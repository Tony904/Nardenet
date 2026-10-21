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
		for (size_t a = 0; a < l->n_inputs; a++) {
			layer_input input = l->inputs[a];
			float* inl_output = &input.data[b * input.stride];
			size_t i;
#pragma omp parallel for
			for (i = 0; i < input.n; i++) {
				Z[i] = inl_output[i];
			}
			Z += input.n;
		}
	}
	if (l->activation) l->activate(Z0, l->output, l->out.n, batch_size);
	if (net->training) zero_array(l->grads, l->out.n * batch_size);
}

void backward_route(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->grads;
	if (l->activation) get_activation_grads(l, batch_size);
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t a = 0; a < l->n_inputs; a++) {
			layer_input input = l->inputs[a];
			float* inl_grads = &input.grads[b * input.stride];
			size_t i;
#pragma omp parallel for
			for (i = 0; i < input.n; i++) {
				inl_grads[i] += grads[i];
			}
			grads += input.n;
		}
	}
}

#ifdef GPU
void forward_route_gpu(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* Z = l->gpu.Z;
	float* Z0 = Z;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t a = 0; a < l->n_inputs; a++) {
			layer_input input = l->inputs[a];
			float* inl_output = &input.gpu.data[b * input.stride];
			copy_array_gpu(inl_output, Z, (int)input.n);
			Z += input.n;
		}
	}
	
	if (l->activation) l->activate(Z0, l->gpu.output, l->out.n, batch_size);
	if (net->training) zero_array_gpu(l->gpu.grads, (int)(l->out.n * batch_size));
}

void backward_route_gpu(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->gpu.grads;
	if (l->activation) get_activation_grads_gpu(l, (int)batch_size);
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t a = 0; a < l->n_inputs; a++) {
			layer_input input = l->inputs[a];
			float* inl_grads = &input.gpu.grads[b * input.stride];
			add_arrays_gpu(grads, inl_grads, (int)input.n);
			grads += input.n;
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