#include "layer_conv.h"
#include <omp.h>
#include "xallocs.h"
#include "im2col.h"
#include "gemm.h"
#include "network.h"
#include "activations.h"


void add_biases(float* output, float* biases, int M, int N);


void forward_layer_first(layer* l, network* net) {
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
}

void forward_layer_conv(layer* l, network* net) {
	int M = (int)(l->n_filters);
	int N = (int)(l->out_w * l->out_h);
	int K = (int)(l->ksize * l->ksize * l->c);
	float* A = l->weights.a;
	float* B = (float*)xcalloc(l->out_n * l->ksize * l->ksize, sizeof(float));
	float* B0 = B;
	float* C = l->output;
	// I think this will work...?
	for (int i = 0; i < l->in_ids.n; i++) {
		int w = l->in_layers[i]->out_w;
		int h = l->in_layers[i]->out_h;
		int c = l->in_layers[i]->out_c;
		float* im = l->in_layers[i]->output;
		B = im2col_cpu(im, (int)c, (int)h, (int)w, (int)l->ksize, (int)l->pad, (int)l->stride, B);
	}
	gemm(M, N, K, A, B0, C);
	add_biases(C, l->biases, M, N);
	l->activate(l);
}

void add_biases(float* output, float* biases, int M, int N) {
	// M = # of filters (aka out_c)
	// N = out_w * out_h
#pragma omp parallel for collapse(2)
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			output[m * N + n] += biases[m];
		}
	}
}

void activate_conv_relu(layer* l) {
	float* act_input = l->act_input;
	float* output = l->output;
#pragma omp parallel for
	for (int i = 0; l->out_n; i++) {
		float x = output[i];
		act_input[i] = x;
		output[i] = relu_x(x);
	}
}

void activate_conv_mish(layer* l) {
	float* act_input = l->act_input;
	float* output = l->output;
#pragma omp parallel for
	for (int i = 0; l->out_n; i++) {
		float x = output[i];
		act_input[i] = x;
		output[i] = mish_x(x, 20);
	}
}

void activate_conv_none(layer* l) {
	l;
}

void backward_layer_first(layer* l, network* net) {
	l; net;
}

void backward_layer_conv(layer* l, network* net) {
	l; net;
}