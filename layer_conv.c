#include "layer_conv.h"
#include <omp.h>
#include <assert.h>
#include "xallocs.h"
#include "im2col.h"
#include "gemm.h"
#include "network.h"
#include "activations.h"
#include "xarrays.h"


void forward_first(layer* l, network* net) {
	int M = (int)(l->n_filters);
	int N = (int)(l->out_w * l->out_h);
	int K = (int)(l->ksize * l->ksize * l->c);
	float* A = l->weights.a;
	float* B = (float*)xcalloc(l->out_n * l->ksize * l->ksize, sizeof(float));
	float* C = l->output;
	im2col_cpu(net->input->data, (int)l->c, (int)l->h, (int)l->w, (int)l->ksize, (int)l->pad, (int)l->stride, B);
	gemm(M, N, K, A, B, C);
	add_biases(C, l->biases, M, N);
	l->activate(l);
	xfree(B);
}

#pragma warning(suppress:4100)
void forward_conv(layer* l, network* net) {
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

void activate_conv_relu(layer* l) {
	float* act_input = l->act_input;
	float* output = l->output;
	int i;
#pragma omp parallel for
	for (i = 0; i < l->out_n; i++) {
		float x = output[i];
		act_input[i] = x;
		output[i] = relu_x(x);
	}
}

void activate_conv_mish(layer* l) {
	float* act_input = l->act_input;
	float* output = l->output;
	int i;
#pragma omp parallel for
	for (i = 0; i < l->out_n; i++) {
		float x = output[i];
		act_input[i] = x;
		output[i] = mish_x(x, 20);
	}
}

void activate_conv_logistic(layer* l) {
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

void activate_conv_none(layer* l) {
	l;
}

void backprop_first(layer* l, network* net) {
	l; net;
}

void backprop_conv(layer* l, network* net) {
	l; net;
}