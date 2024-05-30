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


void forward_classify(layer* l) {
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
	xfree(B0);
	add_biases(C, l->biases, M, N);
	l->activate(l);
	l->get_cost(l);
}

void backprop_classify(layer* l, network* net) {
	if (l->activation == ACT_SOFTMAX) {
		// calculate gradients of logits wrt weights
	}
}
