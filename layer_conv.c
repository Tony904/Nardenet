#include "layer_conv.h"
#include <omp.h>
#include <assert.h>
#include "xallocs.h"
#include "im2col.h"
#include "gemm.h"
#include "network.h"
#include "activations.h"
#include "derivatives.h"
#include "xarrays.h"


void forward_conv(layer* l) {
	int M = (int)(l->n_filters);
	int N = (int)(l->out_w * l->out_h);
	int K = (int)(l->ksize * l->ksize * l->c);
	float* A = l->weights.a;
	float* B = (float*)xcalloc((size_t)(N * K), sizeof(float));
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
	xfree(B0);
}

void backprop_conv(layer* l, network* net) {
	float* grads = l->grads;  // propogated gradients up to this layer
	float* lgrads = (float*)xcalloc(l->out_n, sizeof(float));
	// dz/dw = previous layer input
	// da/dz = mish derivative
	float* Z = l->act_input;
	get_grads_mish(grads, Z, l->out_n);  // dC/da * da/dz
	// grads[] is now dC/dz.
	// now get dz/dw for each weight (it's just the input from the previous array in the forward pass).
	// im2col the input matrix?
}