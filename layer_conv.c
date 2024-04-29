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

void add_biases(float* output, float* biases, int M, int N) {
	// M = # of filters (aka out_c)
	// N = out_w * out_h
#pragma omp parallel for
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
		act_input[i] = output[i];
		output[i] = relu_x(output[i]);
	}
}

void activate_conv_mish(layer* l) {
	float* act_input = l->act_input;
	float* output = l->output;
#pragma omp parallel for
	for (int i = 0; l->out_n; i++) {
		act_input[i] = output[i];
		output[i] = mish_x(output[i], 20);
	}
}

void activate_conv_none(layer* l) {
	l;
}

void forward_layer_conv(layer* l, network* net) {
	l; net;
}

void backward_layer_first(layer* l, network* net) {
	l; net;
}

void backward_layer_conv(layer* l, network* net) {
	l; net;
}