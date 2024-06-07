#include "costs.h"
#include <math.h>


void cost_mse(layer* l) {
	float* grads = l->grads;
	float* errors = l->errors;
	float* output = l->output;
	float* truth = l->truth;
	size_t size = l->n_classes;
#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		float delta = truth[i] - output[i];
		errors[i] = delta * delta;
		grads[i] = delta;  // is not 2 * delta because it doesn't matter apparently despite d/dx(x^2) == 2x
	}
}

// Gradients for softmax are calculated wrt its logits, not its outputs, when paired with cross-entropy loss
// because math. Otherwise we'd have to calculate the Jacobian of the softmax function which is a lot of math.
// https://stackoverflow.com/questions/58461808/understanding-backpropagation-with-softmax
void cost_softmax_cce(layer* l) {
	float* grads = l->grads;
	float* errors = l->errors;
	float* output = l->output;
	float* truth = l->truth;
	float cost = 0;
	int n = (int)l->n_classes;
	int i;
#pragma omp parallel for
	for (i = 0; i < n; ++i) {
		float t = truth[i];
		float p = output[i];
		errors[i] = (t) ? -log(p) : 0;
		grads[i] = t - p;  // This is the dC/da * da/dz for softmax with cross entropy
		cost += errors[i];
	}
	l->cost = cost;
}

void cost_sigmoid_cce(layer* l) {
	float* grads = l->grads;
	float* errors = l->errors;
	float* output = l->output;
	float* truth = l->truth;
	int n = (int)l->n_classes;
	int i;
	for (i = 0; i < n; ++i) {
		float t = truth[i];
		float p = output[i];
		errors[i] = -t * log(p) - (1 - t) * log(1 - p);
		grads[i] = t - p;
	}
}

void cost_bce(layer* l) {
	l;
}
