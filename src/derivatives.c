#include "derivatives.h"
#include "activations.h"


void get_grads_sigmoid(float* grads, float* act_output, size_t out_n, size_t batch_size) {
	size_t n = out_n * batch_size;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		grads[i] *= act_output[i] * (1.0F - act_output[i]);
	}
}

// implementation from Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L26-L31
void get_grads_mish(float* grads, float* act_inputs, size_t out_n, size_t batch_size) {
	size_t n = out_n * batch_size;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		float inp = act_inputs[i];
		const float sp = softplus_x(inp, MISH_THRESH);
		const float grad_sp = 1.0F - exp(-sp);
		const float tsp = tanh_x(sp);
		const float grad_tsp = (1.0F - tsp * tsp) * grad_sp;
		const float grad = inp * grad_tsp + tsp;
		grads[i] *= grad;
	}
}

void get_grads_relu(float* grads, float* act_inputs, size_t out_n, size_t batch_size) {
	size_t n = out_n * batch_size;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		grads[i] *= (act_inputs[i] > 0.0F);
	}
}

void get_grads_leaky_relu(float* grads, float* act_inputs, size_t out_n, size_t batch_size) {
	size_t n = out_n * batch_size;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		grads[i] *= (act_inputs[i] > 0.0F) ? 1.0F : 0.1F;
	}
}

void get_grads_tanh(float* grads, float* act_inputs, size_t out_n, size_t batch_size) {
	size_t n = out_n * batch_size;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		float x = tanh_x(act_inputs[i]);
		grads[i] *= 1 - (x * x);
	}
}

void regularize_l1(float* weight_grads, float* weights, size_t size, float decay) {
	size_t i;
#pragma omp parallel for firstprivate(decay)
	for (i = 0; i < size; i++) {
		weight_grads[i] -= (weights[i] > 0.0F) ? decay : -decay;
	}
}

void regularize_l2(float* weight_grads, float* weights, size_t size, float decay) {
	size_t i;
#pragma omp parallel for firstprivate(decay)
	for (i = 0; i < size; i++) {
		weight_grads[i] -= weights[i] * decay;  // not multiplying by 2 cus it doesn't matter, just set decay to a higher value
	}
}

#pragma warning(suppress:4100)  // unreferenced formal parameter
void regularize_none(float* weight_grads, float* weights, size_t size, float decay) {
}