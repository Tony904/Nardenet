#include "layer_classify.h"
#include <omp.h>
#include <assert.h>
#include <stdio.h>
#include "xallocs.h"
#include "im2col.h"
#include "gemm.h"
#include "network.h"
#include "activations.h"
#include "loss.h"
#include "utils.h"
#include "derivatives.h"
#include "layer_conv.h"


void forward_classify(layer* l, network* net) {
	forward_conv(l, net);
	l->get_loss(l);
}

#pragma warning(suppress:4100)  // unreferenced formal parameter: 'net'
void backward_classify(layer* l, network* net) {
	// calculate gradients of logits (z) wrt weights and logits wrt biases
	// dz/dw is just the activation of the previous layer
	// dz/db is always 1

	float* grads = l->grads;  // Size is n_classes (which is also out_n for classify layer)
	
	float* bias_grads = l->bias_grads;
	// bias gradients = dC/dz
	size_t j;
#pragma omp parallel for
	for (j = 0; j < l->out_n; j++) {
		bias_grads[j] += grads[j];  // += because they will be divided by batch count during update step
	}

	// now get dz/dw for each weight (it's just the input from the previous layer in the forward pass).
	int M = (int)l->n_filters;
	int N = (int)(l->ksize * l->ksize * l->c); // filter length
	int K = (int)(l->out_w * l->out_h); // # of patches (is 1 for fully connected layer)

	float* A = grads;  // M * K
	float* B = net->workspace.a;  // N * K
	zero_array(B, (size_t)(N * K));
	float* B0 = B;
	float* C = l->weight_grads;  // M * N

	int w = (int)l->w;
	int h = (int)l->h;
	for (int i = 0; i < l->in_ids.n; i++) {
		layer* inl = l->in_layers[i];
		int c = (int)inl->out_c;
		float* im = inl->output;
		im2col(im, c, h, w, (int)l->ksize, (int)l->pad, (int)l->stride, B);
		B += N * (int)(l->ksize * l->ksize) * c;
	}
	B = B0;
	gemm_atb(M, N, K, A, B, C);
	// C is now dC/dw for all weights. 
	// Note: C array's storage structure is [filter_index * filter_length + filter_weight_index]

	// Now need to create backpropogated "image" for shallower layer(s),
	// so we need to calculate dz/da (dz of this layer wrt da of input layer(s)),
	// which is just the weights. (dz/da = weights of current layer)
	// and then multiply that by dC/dz (which is currently the grads array).
	// Note: Weight gradients DO NOT propagate back, they are just used to update the weights.

	if (l->id == 0) return;

	A = l->weights.a;  // M * N
	B = grads;  // M * K
	C = net->workspace.a;  // N * K
	zero_array(C, (size_t)(N * K));
	gemm_tab(M, N, K, A, B, C);
	// C is now dC/da in col'd form (as in im2col).
	// So now we need to turn this "expanded" form (col) into the form of the dimensions of
	// the output of the input layer (im). We do this using col2im().

	for (int i = 0; i < l->in_ids.n; i++) {
		layer* inl = l->in_layers[i];
		int c = (int)inl->out_c;
		float* im = inl->grads;
		col2im(C, c, h, w, (int)l->ksize, (int)l->pad, (int)l->stride, im);
	}
}

void update_classify(layer* l, network* net) {
	update_conv(l, net);
}