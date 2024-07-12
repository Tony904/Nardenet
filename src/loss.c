#include "loss.h"
#include <math.h>


#pragma warning(suppress:4100)  // unused param
void loss_mse(layer* l, network* net) {
	float* errors = l->errors;
	float* output = l->output;
	float* grads = l->grads;
	float* truth = l->truth;
	size_t size = l->n_classes;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < size; i++) {
		float delta = truth[i] - output[i];
		errors[i] = delta * delta;
		grads[i] = delta;  // is not 2 * delta because it doesn't matter apparently despite d/dx(x^2) == 2x
	}
}

// Gradients for softmax are calculated wrt its logits, not its outputs, when paired with cross-entropy loss
// because math. Otherwise we'd have to calculate the Jacobian of the softmax function which is a lot of math.
// https://stackoverflow.com/questions/58461808/understanding-backpropagation-with-softmax
void loss_softmax_cce(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* errors = l->errors;
	float* output = l->output;
	float* grads = l->grads;
	float* truth = l->truth;
	size_t n = l->n_classes;
	float loss = 0.0F;
	size_t s;
//#pragma omp parallel for reduction(+:loss) firstprivate(n)
	for (s = 0; s < batch_size; s++) {
		size_t offset = s * n;
		for (size_t i = 0; i < n; ++i) {
			size_t index = offset + i;
			float t = truth[index];
			float p = output[index];
			grads[index] = p - t;  // This is the dC/da * da/dz for softmax with cross entropy
			errors[index] = (t) ? -log(p) : 0.0F;  // Only used for reporting performance, is not used for training
			loss += errors[index];
		}
		print_top_class_name(&truth[s * n], n, net->class_names, 0, 0);
		printf(" : ");
		print_top_class_name(&output[s * n], n, net->class_names, 0, 1);
	}
	l->loss = loss / (float)batch_size;
	printf("Avg class loss:      %f\n", l->loss);
}

#pragma warning(suppress:4100)  // unused param
void loss_sigmoid_cce(layer* l, network* net) {
	float* errors = l->errors;
	float* output = l->output;
	float* grads = l->grads;
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

#pragma warning(suppress:4100)  // unused param
void loss_bce(layer* l, network* net) {
	l;
}

void loss_l1(network* net) {
	size_t n_layers = net->n_layers;
	layer* ls = net->layers;
	float loss = 0.0F;
	float decay = net->decay;
	size_t i;
#pragma omp parallel for reduction(+:loss) firstprivate(decay)
	for (i = 0; i < n_layers; i++) {
		float* weights = ls[i].weights.a;
		for (size_t j = 0; j < ls[i].weights.n; j++) {
			float w = weights[j];
			w = (w < 0.0F) ? -w : w;  // w needs to be positive
			loss += w * decay;
		}
	}
	net->loss = loss;
}

void loss_l2(network* net) {
	size_t n_layers = net->n_layers;
	layer* ls = net->layers;
	float loss = 0.0F;
	float decay = net->decay;
	size_t i;
#pragma omp parallel for reduction(+:loss) firstprivate(decay)
	for (i = 0; i < n_layers; i++) {
		float* weights = ls[i].weights.a;
		for (size_t j = 0; j < ls[i].weights.n; j++) {
			float w = weights[j];
			loss += w * w * decay;
		}
	}
	net->loss = loss;
}