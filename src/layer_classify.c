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
}

void backprop_classify(layer* l, network* net) {
	float* truth = net->truth;
	float* output = l->output;
	float* errors = l->errors;
	float* grads = l->grads;
	if (l->cost_type == COST_MSE) {
		get_cost_mse(grads, errors, output, truth, l->out_n);
		get_grads_mse(grads, errors, l->out_n);
	}
	else {
		printf("under construction\n");
		wait_for_key_then_exit();
	}

}
