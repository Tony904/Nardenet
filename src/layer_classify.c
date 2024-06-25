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
	float* B = net->workspace.a;
	float* B0 = B;
	float* C = l->act_input;
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
	l->activate(l);  // sends l->act_input through activation functio and stores in l->output
	l->get_cost(l);
}

#pragma warning(suppress:4100)  // unreferenced formal parameter: 'net'
void backprop_classify(layer* l, network* net) {
	// calculate gradients of logits (z) wrt weights and logits wrt biases
	// dz/dw is just the activation of the previous layer
	// dz/db is always 1

	float* grads = l->grads;  // Size is n_classes (which is also out_n for classify layer)
	
	float* bias_grads = l->bias_grads;
	// bias gradients = dC/dz
	for (size_t i = 0; i < l->out_n; i++) {
		bias_grads[i] += grads[i];  // += because they will be divided by batch count during update step
	}

	// now get dz/dw for each weight (it's just the input from the previous layer in the forward pass).
	int M = (int)l->n_filters;
	int N = (int)(l->ksize * l->ksize * l->c); // filter length
	int K = (int)(l->out_w * l->out_h); // # of patches

	float* A = grads;  // M * K
	float* B = net->workspace.a;  // N * K
	for (int i = 0; i < N * K; i++) { B[i] = 0.0F; }
	float* B0 = B;
	float* C = l->weight_grads;  // M * N

	int w = (int)l->w;
	int h = (int)l->h;
	for (int i = 0; i < l->in_ids.n; i++) {
		layer* inl = l->in_layers[i];
		assert(w == (int)inl->out_w);
		assert(h == (int)inl->out_h);
		int c = (int)inl->out_c;
		float* im = inl->output;
		B = im2col_cpu(im, c, h, w, (int)l->ksize, (int)l->pad, (int)l->stride, B);
	}
	B = B0;
	gemm_atb(M, N, K, A, B, C);
	// C is now dC/dw for all weights. [filter_index * filter_length + filter_weight_index]
	// Now need to create backpropogated "image" for shallower layer(s).
	for (int i = 0; i < N; i++) { B[i] = 0.0F; }
	sum_columns(M, N, C, B);  // something about this ain't right, i just get zeros.
	for (int i = 0; i < l->in_ids.n; i++) {
		layer* inl = l->in_layers[i];
		int c = (int)inl->out_c;
		float* im = inl->output;
		for (int j = 0; j < inl->out_n; j++) { im[j] = 0.0F; }
		wgrads2im_cpu(B, c, h, w, (int)l->ksize, (int)l->pad, (int)l->stride, im);
		print_float_array(im, w * h * c);
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
//#pragma omp parallel for
	for (w = 0; w < n; w++) {
		weights[w] += weight_grads[w] * rate;
	}
}