#include "layer_classify.h"
#include <omp.h>
#include <assert.h>
#include "xallocs.h"
#include "im2col.h"
#include "gemm.h"
#include "network.h"
#include "activations.h"


void softmax_classify(layer* l);


#pragma warning(suppress:4100)
void forward_classify(layer* l, network* net) {
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
	softmax_classify(l);
}

void activate_classify(layer* l) {
	float* act_input = l->act_input;
	float* output = l->output;
	int i;
#pragma omp parallel for
	for (i = 0; i < l->out_n; i++) {
		float x = output[i];
		act_input[i] = x;
		output[i] = logistic_x(x);
	}
}

void backprop_classify(layer* l, network* net) {
	l; net;
}

void softmax_classify(layer* l) {
	float esum = 0;
	for (int i = 0; i < l->out_n; i++) {
		expf(l->output[i]);
	}
}