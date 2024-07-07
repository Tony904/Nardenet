#include "derivatives.h"
#include "activations.h"


void get_grads_sigmoid(float* grads, float* act_output, size_t size) {
	int n = (int)size;
	int i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		grads[i] *= act_output[i] * (1.f - act_output[i]);
	}
}

// implementation from Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L26-L31
void get_grads_mish(float* grads, float* act_input, size_t size) {
	int n = (int)size;
	int i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		float inp = act_input[i];
		const float sp = softplus_x(inp, MISH_THRESH);
		const float grad_sp = 1 - exp(-sp);
		const float tsp = tanh_x(sp);
		const float grad_tsp = (1 - tsp * tsp) * grad_sp;
		const float grad = inp * grad_tsp + tsp;
		grads[i] *= grad;
	}
}

void get_grads_relu(float* grads, float* act_input, size_t size) {
	size_t i;
#pragma omp parallel for
	for (i = 0; i < size; i++) {
		grads[i] *= (act_input[i] > 0);
	}
}

void get_grads_leaky_relu(float* grads, float* act_input, size_t size) {
	size_t i;
#pragma omp parallel for
	for (i = 0; i < size; i++) {
		grads[i] *= (act_input[i] > 0.0F) ? 1.0F : 0.1F;
	}
}

void regularize_l1(float* weight_grads, float* weights, size_t size, float decay) {
	size_t i;
#pragma omp parallel for firstprivate(decay)
	for (i = 0; i < size; i++) {
		weight_grads[i] -= ((weights[i] > 0.0F) ? decay : -decay);
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
	decay;
}