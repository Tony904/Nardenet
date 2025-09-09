#include "loss.h"
#include <math.h>
#include "xcuda.h"
#include "iou.h"


void loss_mae(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* errors = l->errors;
	float* output = l->output;
	float* grads = l->grads;
	float* truth = l->truth;
	size_t n = l->n_filters;
	float loss = 0.0F;
	size_t b;
#pragma omp parallel for reduction(+:loss) firstprivate(n)
	for (b = 0; b < batch_size; b++) {
		size_t offset = b * n;
		for (size_t i = 0; i < n; ++i) {
			size_t index = offset + i;
			float delta = output[index] - truth[index];
			errors[index] = delta;
			grads[index] = (delta > 0.0F) ? 1.0F : (delta < 0.0F) ? -1.0F : 0.0F;
			loss += errors[index];
		}
	}
	l->loss = loss / (float)batch_size;
}
#pragma warning (suppress:4100)
void loss_mae_gpu(layer* l, network* net) {
#ifdef GPU
	launch_loss_mae_kernel(l->grads, l->output, l->truth, l->errors, (int)l->n, (int)net->batch_size);
#else
	gpu_not_defined();
#endif
}


void loss_mse(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* errors = l->errors;
	float* output = l->output;
	float* grads = l->grads;
	float* truth = l->truth;
	size_t n = l->n_filters;
	float loss = 0.0F;
	size_t b;
#pragma omp parallel for reduction(+:loss) firstprivate(n)
	for (b = 0; b < batch_size; b++) {
		size_t offset = b * n;
		for (size_t i = 0; i < n; ++i) {
			size_t index = offset + i;
			float delta = output[index] - truth[index];
			errors[index] = delta * delta;
			grads[index] = delta;  // is not 2 * delta because it doesn't matter despite d/dx(x^2) == 2x, just double the learning rate to match it
			loss += errors[index];
		}
	}
	l->loss = loss / (float)batch_size;
}
#pragma warning (suppress:4100)
void loss_mse_gpu(layer* l, network* net) {
#ifdef GPU
	launch_loss_mse_kernel(l->grads, l->output, l->truth, l->errors, (int)l->n, (int)net->batch_size);
#else
	gpu_not_defined();
#endif
}


// Gradients for softmax are calculated wrt its logits, not its outputs, when paired with cross-entropy loss
// because math. Otherwise we'd have to calculate the Jacobian of the softmax function which is a lot of math.
// https://stackoverflow.com/questions/58461808/understanding-backpropagation-with-softmax
void loss_cce(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* errors = l->errors;
	float* output = l->output;
	float* grads = l->grads;
	float* truth = l->truth;
	size_t n = l->n_classes;
	float loss = 0.0F;
	size_t b;
#pragma omp parallel for reduction(+:loss) firstprivate(n)
	for (b = 0; b < batch_size; b++) {
		size_t offset = b * n;
		for (size_t i = 0; i < n; ++i) {
			size_t index = offset + i;
			float t = truth[index];
			float p = output[index];
			grads[index] = p - t;  // This is the dC/da * da/dz for softmax with cross entropy
			errors[index] = (t) ? -logf(p) : 0.0F;  // Errors are only used for reporting performance, not for backprop
			loss += errors[index];
		}
	}
	l->loss = loss / (float)batch_size;
}
#pragma warning (suppress:4100)
void loss_cce_gpu(layer* l, network* net) {
#ifdef GPU
	launch_loss_cce_kernel(l->grads, l->output, l->truth, l->errors, (int)l->n, (int)net->batch_size);
#else
	gpu_not_defined();
#endif
}


// binary cross entropy
void loss_bce(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* errors = l->errors;
	float* output = l->output;
	float* grads = l->grads;
	float* truth = l->truth;
	size_t n = l->n_filters;
	float loss = 0.0F;
	size_t b;
#pragma omp parallel for reduction(+:loss) firstprivate(n)
	for (b = 0; b < batch_size; b++) {
		size_t offset = b * n;
		for (size_t i = 0; i < n; ++i) {
			size_t index = offset + i;
			float t = truth[index];
			float p = output[index];
			grads[index] = p - t;  // dC/da
			errors[index] = -t * logf(p) - (1.0F - t) * logf(1.0F - p);
			loss += errors[index];
		}
	}
	l->loss = loss / (float)batch_size;
}
#pragma warning (suppress:4100)
void loss_bce_gpu(layer* l, network* net) {
#ifdef GPU
	launch_loss_bce_kernel(l->grads, l->output, l->truth, l->errors, (int)l->n, (int)net->batch_size);
#else
	gpu_not_defined();
#endif
}


void loss_l1(network* net) {
	size_t n_layers = net->n_layers;
	layer* ls = net->layers;
	float loss = 0.0F;
	float decay = net->decay;
	size_t i;
#pragma omp parallel for reduction(+:loss) firstprivate(decay)
	for (i = 0; i < n_layers; i++) {
		float* weights = ls[i].weights;
		for (size_t j = 0; j < ls[i].n_weights; j++) {
			float w = weights[j];
			if (w < 0.0F) w = -w;  // w needs to be positive
			loss += w * decay;
		}
	}
	net->loss = loss;
}
void loss_l1_gpu(network* net) {
	net;
	// WILL DO LATER... PROBABLY...
}

void loss_l2(network* net) {
	size_t n_layers = net->n_layers;
	layer* ls = net->layers;
	float loss = 0.0F;
	float decay = net->decay;
	size_t i;
#pragma omp parallel for reduction(+:loss) firstprivate(decay)
	for (i = 0; i < n_layers; i++) {
		float* weights = ls[i].weights;
		for (size_t j = 0; j < ls[i].n_weights; j++) {
			float w = weights[j];
			loss += w * w * decay;
		}
	}
	net->loss = loss;
}
void loss_l2_gpu(network* net) {
	net;
	// WILL DO LATER... PROBABLY...
}