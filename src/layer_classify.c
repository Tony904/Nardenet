#include "layer_classify.h"
#include <omp.h>
#include <assert.h>
#include <stdio.h>
#include "xallocs.h"
#include "im2col.h"
#include "gemm.h"
#include "network.h"
#include "activations.h"
#include "costs.h"
#include "utils.h"
#include "derivatives.h"


void forward_classify(layer* l, network* net) {
	int M = (int)(l->n_filters);
	int N = (int)(l->out_w * l->out_h);
	int K = (int)(l->ksize * l->ksize * l->c);
	float* A = l->weights.a;
	//float* B = (float*)xcalloc((size_t)(N * K), sizeof(float));
	float* B = net->workspace.a;
	float* B0 = B;
	float* C = l->output;
	int w = (int)l->w;
	int h = (int)l->h;
	for (int i = 0; i < l->in_ids.n; i++) {
		assert(w == (int)l->in_layers[i]->out_w);
		assert(h == (int)l->in_layers[i]->out_h);
		int c = (int)l->in_layers[i]->out_c;
		float* im = l->in_layers[i]->output;
		B = im2col_cpu(im, c, h, w, (int)l->ksize, (int)l->pad, (int)l->stride, B);
	}
	gemm(M, N, K, A, B0, C);
	add_biases(C, l->biases, M, N);
	l->activate(l);
	l->get_cost(l);
}

#pragma warning(suppress:4100)  // unreferenced formal parameter: 'net'
void backprop_classify(layer* l, network* net) {
	// calculate gradients of logits wrt weights and logits wrt biases
	// dz/dw is just the activation of the previous layer
	// dz/db is always 1
	float* weight_grads = l->weight_grads;
	float* grads = l->grads;
	float* bias_grads = l->bias_grads;
	// bias gradients = dC/dz
	for (size_t i = 0; i < l->out_n; i++) {
		bias_grads[i] += grads[i];  // += because they will be divided by batch count during update step
	}
	size_t w = 0;
	int i;
#pragma omp parallel for
	for (i = 0; i < l->in_ids.n; i++) {
		layer* inl = l->in_layers[i];
		float* inl_grads = inl->grads;
		float* output = inl->output;
		for (size_t j = 0; j < inl->out_n; j++) {
			float grad = grads[w] * output[j];  // dC/dw = dC/dz * dz/dw
			inl_grads[j] = grad;
			weight_grads[w] += grad;  // += because they will be divided by batch count during update step
			++w;
		}
	}
}

void update_classify(layer* l, network* net) {
	float batch_size = (float)net->batch_size;
	float rate = net->learning_rate * batch_size;

	float* biases = l->biases;
	float* bias_grads = l->bias_grads;
	int n = (int)l->n_filters;
	for (int b = 0; b < n; b++) {
		biases[b] += bias_grads[b] * rate;
	}

	float* weights = l->weights.a;
	float* weight_grads = l->weight_grads;
	n = (int)l->weights.n;
	int w;
#pragma omp parallel for
	for (w = 0; w < n; w++) {
		weights[w] += weight_grads[w] * rate;
	}
}