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

#pragma warning(disable:4100)
#pragma warning(disable:4189)

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
	xfree(B0);
}

void backprop_conv(layer* l, network* net) {
	float* grads = l->grads;  // propogated gradients up to this layer
	// dz/dw = previous layer (shallower layer) input
	// da/dz = mish derivative
	float* Z = l->act_input;
	get_grads_mish(grads, Z, l->out_n);  // dC/da * da/dz
	// grads[] is now dC/dz.
	// now get dz/dw for each weight (it's just the input from the previous layer in the forward pass).
	
	int M = (int)l->n_filters;
	int N = (int)(l->ksize * l->ksize * l->c); // filter length
	int K = (int)(l->out_n); // # of patches

	float* A = grads;  // M * K
	float* B = (float*)xcalloc((size_t)(N * K), sizeof(float));
	float* B0 = B;
	float* C = l->weight_updates;  // M * N

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
	gemm_atb(M, N, K, A, B0, C);
	// C is now dC/dw for all weights. [filter_index * filter_length + filter_weight_index]
	// Now need to create backpropogated "image" for shallower layer(s). col2im?


}