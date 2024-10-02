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
#include "batchnorm.h"


void forward_conv(layer* l, network* net) {
	size_t out_n = l->out_n;
	size_t batch_size = net->batch_size;
	zero_array(l->Z, out_n * batch_size);
	size_t n_groups = l->n_groups;
	size_t w = l->w;
	size_t h = l->h;
	size_t M = l->n_filters;
	size_t N = l->out_w * l->out_h;
	size_t K = l->ksize * l->ksize * l->c;
	float* A = l->weights.a;  // M * K
	for (size_t b = 0; b < batch_size; b++) {
		float* B = net->workspace.a;  // K * N
		float* B0 = B;
		for (size_t i = 0; i < l->in_ids.n; i++) {
			layer* inl = l->in_layers[i];
			size_t c = inl->out_c;
			float* im = &inl->output[b * inl->out_n];
			im2col(im, (int)c, (int)h, (int)w, (int)l->ksize, (int)l->pad, (int)l->stride, B);
			B += K * l->ksize * l->ksize * c;
		}
		float* C = &l->Z[b * M * N];  // M * N
		if (n_groups > 1) gemm_groups(M, N, K, A, B0, C, n_groups);
		else gemm(M, N, K, A, B0, C);
	}
	if (l->batch_norm) {
		forward_batch_norm(l, batch_size);
		l->activate(l->act_inputs, l->output, out_n, batch_size);
	}
	else {
		// note: l->Z = l->act_inputs when batchnorm disabled
		add_biases(l->Z, l->biases, M, N, batch_size);
		l->activate(l->Z, l->output, out_n, batch_size);
	}
	if (net->training) zero_array(l->grads, out_n * batch_size);
}

void backward_conv(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->grads;  // propogated gradients up to this layer
	// dz/dw = previous layer (shallower layer) input
	// da/dz = activation derivative
	if (l->activation == ACT_MISH) get_grads_mish(grads, l->act_inputs, l->out_n, batch_size);  // dC/da * da/dz
	else if (l->activation == ACT_RELU) get_grads_relu(grads, l->act_inputs, l->out_n, batch_size);
	else if (l->activation == ACT_LEAKY) get_grads_leaky_relu(grads, l->act_inputs, l->out_n, batch_size);
	else if (l->activation == ACT_SIGMOID) get_grads_sigmoid(grads, l->output, l->out_n, batch_size);
	else if (l->activation == ACT_SOFTMAX);
	else if (l->activation == ACT_TANH) get_grads_tanh(grads, l->act_inputs, l->out_n, batch_size);
	else if (l->activation == ACT_NONE);
	else {
		printf("Incorrect or unsupported activation function.\n");
		wait_for_key_then_exit();
	}
	// grads[] is now dC/dz.
	
	// now get dz/dw for each weight (it's just the input from the previous layer in the forward pass).
	size_t M = l->n_filters;
	size_t N = l->ksize * l->ksize * l->c; // filter length
	size_t K = l->out_w * l->out_h; // # of patches

	// sum dC/dz for each filter to get it's bias gradients.
	get_bias_grads(l->bias_grads, grads, M, K, batch_size);  // note: biases = betas for batch norm

	if (l->batch_norm) backward_batch_norm(l, batch_size);

	size_t n_groups = l->n_groups;
	size_t w = l->w;
	size_t h = l->h;
	for (size_t s = 0; s < batch_size; s++) {
		float* A = &grads[s * M * K];  // M * K
		float* B = net->workspace.a;  // N * K
		zero_array(B, (size_t)(N * K));
		float* B0 = B;
		float* C = l->weight_grads;  // M * N
		for (size_t i = 0; i < l->in_ids.n; i++) {
			layer* inl = l->in_layers[i];
			size_t c = inl->out_c;
			float* im = &inl->output[s * inl->out_n];
			im2col(im, (int)c, (int)h, (int)w, (int)l->ksize, (int)l->pad, (int)l->stride, B);
			B += N * l->ksize * l->ksize * c;
		}
		B = B0;
		if (n_groups > 1) gemm_atb_groups(M, N, K, A, B, C, n_groups);
		else gemm_atb(M, N, K, A, B, C);
	}
	// C is now dC/dw for all weights. 
	// Note: C array's storage structure is [filter_index * filter_length + filter_weight_index]
	
	// Now need to create backpropogated "image" for shallower layer(s),
	// so we need to calculate dz/da (dz of this layer wrt da of input layer(s)),
	// which is just the weights. (dz/da = weights of current layer)
	// and then multiply that by dC/dz (which is currently the grads array).
	// Note: Weight gradients DO NOT propagate back, they are just used to update the weights.

	if (l->id == 0) return;
	for (size_t s = 0; s < batch_size; s++) {
		float* A = l->weights.a;  // M * N
		float* B = &grads[s * M * K];  // M * K
		float* C = net->workspace.a;  // N * K
		zero_array(C, N * K);
		if (n_groups > 1) gemm_tab_groups(M, N, K, A, B, C, n_groups);
		else gemm_tab(M, N, K, A, B, C);
		// C is now dC/da in col'd form (as in im2col).
		// So now we need to turn this "expanded" form (col) into the form of the dimensions of
		// the output of the input layer (im). We do this using col2im().

		for (size_t i = 0; i < l->in_ids.n; i++) {
			layer* inl = l->in_layers[i];
			size_t c = inl->out_c;
			float* im = &inl->grads[s * w * h * c];
			col2im(C, (int)c, (int)h, (int)w, (int)l->ksize, (int)l->pad, (int)l->stride, im);
		}
	}
}

void update_conv(layer* l, network* net) {
	float rate = net->current_learning_rate;	
	float momentum = net->momentum;

	float* biases = l->biases;
	float* bias_grads = l->bias_grads;
	float* biases_velocity = l->biases_velocity;
	size_t n = l->n_filters;
	size_t i;
#pragma omp parallel for firstprivate(rate, momentum)
	for (i = 0; i < n; i++) {
		float v_old = biases_velocity[i];
		float v_new = momentum * v_old - rate * bias_grads[i];
		biases[i] += -momentum * v_old + (1 + momentum) * v_new;  // Nesterov momentum
		biases_velocity[i] = v_new;
		bias_grads[i] = 0.0F;
	}

	float* weights = l->weights.a;
	float* weight_grads = l->weight_grads;
	float* weights_velocity = l->weights_velocity;
	n = l->weights.n;
	net->regularize_weights(weight_grads, weights, n, net->decay);
#pragma omp parallel for firstprivate(rate, momentum)
	for (i = 0; i < n; i++) {
		float v_old = weights_velocity[i];
		float v_new = momentum * v_old - rate * weight_grads[i];
		weights[i] += -momentum * v_old + (1 + momentum) * v_new;
		weights_velocity[i] = v_new;
		weight_grads[i] = 0.0F;
	}

	if (l->batch_norm) {
		float* gammas = l->gammas;
		float* gamma_grads = l->gamma_grads;
		float* gammas_velocity = l->gammas_velocity;
		n = l->out_c;
#pragma omp parallel for firstprivate(rate, momentum)
		for (i = 0; i < n; i++) {
			float v_old = weights_velocity[i];
			float v_new = momentum * v_old - rate * weight_grads[i];
			gammas[i] += -momentum * v_old + (1 + momentum) * v_new;
			gammas_velocity[i] = v_new;
			gamma_grads[i] = 0.0F;
		}
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
	l->Z = (float*)xcalloc(l->n_filters * l->out_w * l->out_h, sizeof(float));
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

	pprint_mat(l->Z, (int)l->out_w, (int)l->out_h, (int)l->out_c);
}