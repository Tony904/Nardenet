#include "layer_conv.h"
#include <omp.h>
#include <assert.h>
#include <string.h>
#include "xallocs.h"
#include "im2col.h"
#include "gemm.h"
#include "network.h"
#include "activations.h"
#include "derivatives.h"
#include "xarrays.h"


void forward_conv(layer* l, network* net) {
	int M = (int)(l->n_filters);
	int N = (int)(l->out_w * l->out_h);
	int K = (int)(l->ksize * l->ksize * l->c);
	float* A = l->weights.a;
	//float* B = (float*)xcalloc((size_t)(N * K), sizeof(float));
	float* B = net->workspace.a;
	float* B0 = B;
	float* C = l->output;
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
	gemm(M, N, K, A, B0, C);
	add_biases(C, l->biases, M, N);
	l->activate(l);
	//xfree(B0);
}

void backprop_conv(layer* l, network* net) {
	float* grads = l->grads;  // propogated gradients up to this layer
	// dz/dw = previous layer (shallower layer) input
	// da/dz = activation derivative
	float* Z = l->act_input;
	if (l->activation == ACT_MISH) get_grads_mish(grads, Z, l->out_n);  // dC/da * da/dz
	else {
		printf("TODO: IMPLEMENT OTHER ACTIVATION GRADIENT FUNCTIONS.\n");
		exit(EXIT_FAILURE);
	}
	// grads[] is now dC/dz.
	// sum dC/dz for each filter to get it's bias gradients.
	float* bias_grads = l->bias_grads;
	int out_wh = (int)(l->out_w * l->out_h);
	int omp_i;
#pragma omp parallel for
	for (omp_i = 0; omp_i < l->n_filters; omp_i++) {
		float sum = 0;
		int iwh = omp_i * out_wh;
		for (int omp_j = 0; omp_j < out_wh; omp_j++) {
			sum += grads[iwh + omp_j];
		}
		bias_grads[omp_i] += sum;  // += because they will be divided by batch count during update step
	}

	// now get dz/dw for each weight (it's just the input from the previous layer in the forward pass).
	int M = (int)l->n_filters;
	int N = (int)(l->ksize * l->ksize * l->c); // filter length
	int K = (int)(l->out_w * l->out_h); // # of patches

	float* A = grads;  // M * K
	//float* B = (float*)xcalloc((size_t)(N * K), sizeof(float));
	float* B = net->workspace.a;
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
	sum_columns(M, N, C, B);
	printf("sum_columns done.\n");
	for (int i = 0; i < l->in_ids.n; i++) {
		layer* inl = l->in_layers[i];
		int c = (int)inl->out_c;
		float* im = inl->output;
		for (int j = 0; j < inl->out_n; j++) { im[j] = 0.0F; }
		printf("wgrads2im, in_layer: %d\n", i);
		wgrads2im_cpu(B, c, h, w, (int)l->ksize, (int)l->pad, (int)l->stride, im);
	}
}

void update_conv(layer* l, network* net) {
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
#pragma omp parallel for
	for (w = 0; w < n; w++) {
		weights[w] += weight_grads[w] * rate;
	}
}