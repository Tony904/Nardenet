#include "layer_classify.h"
#include <omp.h>
#include "xallocs.h"
#include "im2col.h"
#include "gemm.h"
#include "network.h"
#include "activations.h"

#pragma warning(suppress:4100)
void forward_layer_classify(layer* l, network* net) {
	int M = (int)(l->n_filters);
	int N = (int)(l->out_w * l->out_h);
	int K = (int)(l->ksize * l->ksize * l->c);
	float* A = l->weights.a;
	float* B = (float*)xcalloc((size_t)(N * K), sizeof(float));
	float* B0 = B;
	float* C = l->output;
	// I think this will work...?
	for (int i = 0; i < l->in_ids.n; i++) {
		int w = (int)l->in_layers[i]->out_w;
		int h = (int)l->in_layers[i]->out_h;
		int c = (int)l->in_layers[i]->out_c;
		float* im = l->in_layers[i]->output;
		B = im2col_cpu(im, (int)c, (int)h, (int)w, (int)l->ksize, (int)l->pad, (int)l->stride, B);
	}
	gemm(M, N, K, A, B0, C);
	add_biases(C, l->biases, M, N);
	l->activate(l);
	xfree(B0);
}

void backward_layer_classify(layer* l, network* net) {
	l; net;
}

void activate_classify(layer* l) {
	l;
}