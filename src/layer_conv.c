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
#include "utils.h"


void forward_conv(layer* l, network* net) {
	int M = (int)(l->n_filters);
	int N = (int)(l->out_w * l->out_h);
	int K = (int)(l->ksize * l->ksize * l->c);
	float* A = l->weights.a;  // M * K
	float* B = net->workspace.a;  // K * N
	float* B0 = B;
	float* C = l->act_input;  // M * N
	zero_array(C, (size_t)(M * N));
	int w = (int)l->w;
	int h = (int)l->h;
	for (int i = 0; i < l->in_ids.n; i++) {
		layer* inl = l->in_layers[i];
		assert(w == (int)inl->out_w);
		assert(h == (int)inl->out_h);
		int c = (int)inl->out_c;
		float* im = inl->output;
		im2col(im, c, h, w, (int)l->ksize, (int)l->pad, (int)l->stride, B);
		B += K * (int)(l->ksize * l->ksize) * c;
	}
	gemm(M, N, K, A, B0, C);
	add_biases(C, l->biases, M, N);
	l->activate(l);  // sends l->act_input through activation function and stores in l->output
}

void backward_conv(layer* l, network* net) {
	float* grads = l->output;  // propogated gradients up to this layer
	// dz/dw = previous layer (shallower layer) input
	// da/dz = activation derivative
	float* Z = l->act_input;
	if (l->activation == ACT_MISH) get_grads_mish(grads, Z, l->out_n);  // dC/da * da/dz
	else if (l->activation == ACT_RELU) get_grads_relu(grads, Z, l->out_n);
	else if (l->activation == ACT_LEAKY) get_grads_leaky_relu(grads, Z, l->out_n);
	else {
		printf("Incorrect or unsupported activation function.\n");
		exit(EXIT_FAILURE);
	}
	// grads[] is now dC/dz.
	
	// now get dz/dw for each weight (it's just the input from the previous layer in the forward pass).
	int M = (int)l->n_filters;
	int N = (int)(l->ksize * l->ksize * l->c); // filter length
	int K = (int)(l->out_w * l->out_h); // # of patches

	// sum dC/dz for each filter to get it's bias gradients.
	get_bias_grads(l->bias_grads, l->grads, M, K);

	float* A = grads;  // M * K
	float* B = net->workspace.a;  // N * K
	zero_array(B, (size_t)(N * K));
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
		im2col(im, c, h, w, (int)l->ksize, (int)l->pad, (int)l->stride, B);
		B += N * (int)(l->ksize * l->ksize) * c;
	}
	B = B0;
	gemm_atb(M, N, K, A, B, C);
	// C is now dC/dw for all weights. 
	// Note: C array's storage structure is [filter_index * filter_length + filter_weight_index]
	
	// Now need to create backpropogated "image" for shallower layer(s),
	// so we need to calculate dz/da (dz of this layer wrt da of input layer(s)),
	// which is just the weights. (dz/da = weights of current layer)
	// and then multiply that by dC/dz (which is currently the grads array).
	// Note: Weight gradients DO NOT propagate back, they are just used to update the weights.

	if (l->id == 0) return;

	A = l->weights.a;  // M * N
	B = grads;  // M * K
	C = net->workspace.a;  // N * K
	zero_array(C, (size_t)(N * K));
	gemm_tab(M, N, K, A, B, C);
	// C is now dC/da in col'd form (as in im2col).
	// So now we need to turn this "expanded" form (col) into the form of the dimensions of
	// the output of the input layer (im). We do this using col2im().
	
	for (int i = 0; i < l->in_ids.n; i++) {
		layer* inl = l->in_layers[i];
		int c = (int)inl->out_c;
		float* im = inl->output;
		zero_array(im, inl->out_n);
		col2im(C, c, h, w, (int)l->ksize, (int)l->pad, (int)l->stride, im);
	}
}

void update_conv(layer* l, network* net) {
	float rate = net->current_learning_rate;	
	float momentum = net->momentum;

	float* biases = l->biases;
	float* bias_grads = l->bias_grads;
	float* biases_velocity = l->biases_velocity;
	size_t n = l->n_filters;
	size_t b;
#pragma omp parallel for firstprivate(rate, momentum)
	for (b = 0; b < n; b++) {
		float v_old = biases_velocity[b];
		float v_new = momentum * v_old - rate * bias_grads[b];
		biases[b] += -momentum * v_old + (1 + momentum) * v_new;  // Nesterov momentum
		biases_velocity[b] = v_new;
		bias_grads[b] = 0.0F;
	}

	float* weights = l->weights.a;
	float* weight_grads = l->weight_grads;
	float* weights_velocity = l->weights_velocity;
	n = l->weights.n;
	size_t w;
#pragma omp parallel for firstprivate(rate, momentum)
	for (w = 0; w < n; w++) {
		float v_old = weights_velocity[w];
		float v_new = momentum * v_old - rate * weight_grads[w];
		weights[w] += -momentum * v_old + (1 + momentum) * v_new;  // Nesterov momentum
		weights_velocity[w] = v_new;
		weight_grads[w] = 0.0F;
	}
}

/*** TESTS ***/

void test_forward_conv(void) {
	layer* l = (layer*)xcalloc(1, sizeof(layer));
	network* net = (network*)xcalloc(1, sizeof(network));
	l->n_filters = 2;
	l->ksize = 2;
	l->pad = 0;
	l->stride = 1;
	l->w = 3;
	l->h = 3;
	l->c = 3;
	l->out_w = (l->w + 2 * l->pad - l->ksize) / l->stride + 1;
	l->out_h = (l->h + 2 * l->pad - l->ksize) / l->stride + 1;
	l->out_c = l->n_filters;
	l->biases = (float*)xcalloc(l->n_filters, sizeof(float));
	fill_array(l->biases, l->n_filters, 0.5F);
	l->activate = activate_none;
	l->weights.n = l->n_filters * l->ksize * l->ksize * l->c;
	l->weights.a = (float*)xcalloc(l->weights.n, sizeof(float));
	fill_array(l->weights.a, l->weights.n, 1.0F);
	l->act_input = (float*)xcalloc(l->n_filters * l->out_w * l->out_h, sizeof(float));
	layer* inl1 = (layer*)xcalloc(1, sizeof(layer));
	layer* inl2 = (layer*)xcalloc(1, sizeof(layer));
	inl1->out_w = l->w;
	inl2->out_w = l->w;
	inl1->out_h = l->h;
	inl2->out_h = l->h;
	inl1->out_c = 1;
	inl2->out_c = 2;
	assert(l->c == inl1->out_c + inl2->out_c);
	inl1->out_n = inl1->out_w * inl1->out_h * inl1->out_c;
	inl2->out_n = inl2->out_w * inl2->out_h * inl2->out_c;
	inl1->output = (float*)xcalloc(inl1->out_n, sizeof(float));
	fill_array(inl1->output, inl1->out_n, 2.0F);
	inl2->output = (float*)xcalloc(inl2->out_n, sizeof(float));
	fill_array(inl2->output, inl2->out_n, 1.0F);
	l->in_layers = (layer**)xcalloc(2, sizeof(layer*));
	l->in_layers[0] = inl1;
	l->in_layers[1] = inl2;
	l->in_ids.n = 2;

	net->workspace.a = (float*)xcalloc(l->out_w * l->out_h * l->ksize * l->ksize * l->c * 2, sizeof(float));

	forward_conv(l, net);

	pprint_mat(l->act_input, (int)l->out_w, (int)l->out_h, (int)l->out_c);
}