#include "activations.h"


void activate_relu(layer* l) {
	float* act_input = l->act_input;
	float* output = l->output;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < l->out_n; i++) {
		output[i] = relu_x(act_input[i]);
	}
}

void activate_leaky_relu(layer* l) {
	float* act_input = l->act_input;
	float* output = l->output;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < l->out_n; i++) {
		output[i] = leaky_x(act_input[i]);
	}
}

void activate_mish(layer* l) {
	float* act_input = l->act_input;
	float* output = l->output;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < l->out_n; i++) {
		output[i] = mish_x(act_input[i], MISH_THRESH);
	}
}

void activate_sigmoid(layer* l) {
	float* act_input = l->act_input;
	float* output = l->output;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < l->out_n; i++) {
		output[i] = sigmoid_x(act_input[i]);
	}
}

void activate_softmax(layer* l) {
	float* dst = l->output;
	float* src = l->act_input;
	int size = (int)l->out_n;
	float sum = 0.0F;
	// Calculate maxval to then subtract for numerical stability
	// https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
	float maxval = src[0];
	size_t i;
	for (i = 1; i < size; i++) {
		if (src[i] > maxval) {
			maxval = src[i];
		}
	}
	//#pragma omp parallel for
	for (i = 0; i < size; i++) {
		float e = expf(src[i] - maxval);
		sum += e;
		dst[i] = e;
	}
	for (i = 0; i < size; i++) dst[i] /= sum;
}

void activate_none(layer* l) {
	l;
}