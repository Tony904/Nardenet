#include "layer_conv.h"
#include <omp.h>
#include <assert.h>
#include "xallocs.h"
#include "im2col.h"
#include "gemm.h"
#include "activations.h"
#include "derivatives.h"
#include "xarrays.h"
#include "utils.h"
#include "batchnorm.h"
#include "xcuda.h"
#include "blas.h"



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
	float* A = l->weights;  // M * K
	for (size_t b = 0; b < batch_size; b++) {
		float* B = net->workspace;  // K * N
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
	if (l->batchnorm) {
		forward_batchnorm(l, batch_size);
		l->activate(l->act_inputs, l->output, out_n, batch_size);
	}
	else {
		// note: l->Z = l->act_inputs when batchnorm disabled
		add_biases(l->Z, l->biases, M, N, batch_size);
		l->activate(l->Z, l->output, out_n, batch_size);
	}
	if (net->training) zero_array(l->grads, out_n * batch_size);
	print_float_array(l->weights, l->n_weights);
}

void backward_conv(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->grads;  // propogated gradients up to this layer
	// dz/dw = previous layer (shallower layer) input
	// da/dz = activation derivative
	get_activation_grads(l, batch_size); // dC/da * da/dz
	// grads[] is now dC/dz.
	
	// now get dz/dw for each weight (it's just the input from the previous layer in the forward pass).
	size_t M = l->n_filters;
	size_t N = l->ksize * l->ksize * l->c; // filter length
	size_t K = l->out_w * l->out_h; // # of patches

	// sum dC/dz for each filter to get it's bias gradients.
	get_bias_grads(l->bias_grads, grads, M, K, batch_size);  // note: biases = betas for batch norm

	if (l->batchnorm) backward_batchnorm(l, batch_size);
	size_t n_groups = l->n_groups;
	size_t w = l->w;
	size_t h = l->h;
	for (size_t s = 0; s < batch_size; s++) {
		float* A = &grads[s * M * K];  // M * K
		float* B = net->workspace;  // N * K
		zero_array(B, N * K);
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
		float* A = l->weights;  // M * N / n_groups
		float* B = &grads[s * M * K];  // M * K
		float* C = net->workspace;  // N * K
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
	float* bias_velocities = l->bias_velocities;
	size_t n = l->n_filters;
	size_t i;
#pragma omp parallel for firstprivate(rate, momentum)
	for (i = 0; i < n; i++) {
		float v_old = bias_velocities[i];
		float v_new = momentum * v_old - rate * bias_grads[i];
		biases[i] += -momentum * v_old + (1.0F + momentum) * v_new;  // Nesterov momentum
		bias_velocities[i] = v_new;
		bias_grads[i] = 0.0F;
	}
	float* weights = l->weights;
	float* weight_grads = l->weight_grads;
	float* weight_velocities = l->weight_velocities;
	n = l->n_weights;
	net->regularize_weights(weight_grads, weights, n, net->decay);
#pragma omp parallel for firstprivate(rate, momentum)
	for (i = 0; i < n; i++) {
		float v_old = weight_velocities[i];
		float v_new = momentum * v_old - rate * weight_grads[i];
		weights[i] += -momentum * v_old + (1.0F + momentum) * v_new;
		weight_velocities[i] = v_new;
		weight_grads[i] = 0.0F;
	}
	if (l->batchnorm) {
		float* gammas = l->gammas;
		float* gamma_grads = l->gamma_grads;
		float* gamma_velocities = l->gamma_velocities;
		n = l->out_c;
#pragma omp parallel for firstprivate(rate, momentum)
		for (i = 0; i < n; i++) {
			float v_old = weight_velocities[i];
			float v_new = momentum * v_old - rate * weight_grads[i];
			gammas[i] += -momentum * v_old + (1.0F + momentum) * v_new;
			gamma_velocities[i] = v_new;
			gamma_grads[i] = 0.0F;
		}
	}
}

#ifdef GPU
void forward_conv_gpu(layer* l, network* net) {
	int out_n = (int)l->out_n;
	int batch_size = (int)net->batch_size;
	zero_array_gpu(l->gpu.Z, out_n * batch_size);
	int n_groups = (int)l->n_groups;
	int w = (int)l->w;
	int h = (int)l->h;
	int out_w = (int)l->out_w;
	int out_h = (int)l->out_h;
	int ksize = (int)l->ksize;
	int stride = (int)l->stride;
	int pad = (int)l->pad;
	int M = (int)l->n_filters;
	int N = out_w * out_h;
	int K = ksize * ksize * (int)l->c;
	float* A = l->gpu.weights;  // M * K
	for (int b = 0; b < batch_size; b++) {
		float* B = net->gpu.workspace;  // K * N
		float* B0 = B;
		for (int i = 0; i < (int)l->in_ids.n; i++) {
			layer* inl = l->in_layers[i];
			int c = (int)inl->out_c;
			float* im = &inl->gpu.output[b * inl->out_n];
			im2col_gpu(im, B, w, h, c, out_w, out_h, ksize, stride, pad);
			B += K * ksize * ksize * c;
		}
		float* C = &l->gpu.Z[b * M * N];  // M * N
		gemm_gpu(M, N, K, A, B0, C, n_groups);
	}
	if (l->batchnorm) {
		forward_batchnorm_gpu(l->gpu.gammas, l->gpu.biases, l->gpu.means, l->gpu.variances, l->gpu.rolling_means, l->gpu.rolling_variances, l->gpu.Z, l->gpu.Z_norm, l->gpu.act_inputs, w * h, M, batch_size);
		l->activate(l->gpu.act_inputs, l->gpu.output, out_n, batch_size);
	}
	else {
		add_biases_gpu(l->gpu.Z, l->gpu.biases, N, M, batch_size);
		l->activate(l->gpu.Z, l->gpu.output, out_n, batch_size);
	}
	if (net->training) zero_array_gpu(l->gpu.grads, out_n * batch_size);
}

void backward_conv_gpu(layer* l, network* net) {
	int batch_size = (int)net->batch_size;
	float* grads = l->gpu.grads;

	get_activation_grads_gpu(l, batch_size);
	int M = (int)l->n_filters;
	int N = (int)(l->ksize * l->ksize * l->c); // weights per filter
	int K = (int)(l->out_w * l->out_h); // # of patches

	get_bias_grads_gpu(l->gpu.bias_grads, grads, M, K, batch_size);

	if (l->batchnorm) backward_batchnorm_gpu(grads, l->gpu.Z, l->gpu.Z_norm, l->gpu.means, l->gpu.variances, l->gpu.gammas, l->gpu.gamma_grads, K, M, batch_size);

	int n_groups = (int)l->n_groups;
	int w = (int)l->w;
	int h = (int)l->h;
	int out_w = (int)l->out_w;
	int out_h = (int)l->out_h;
	int ksize = (int)l->ksize;
	int stride = (int)l->stride;
	int pad = (int)l->pad;
	for (int s = 0; s < batch_size; s++) {
		float* A = &grads[s * M * K];  // M * K
		float* B = net->gpu.workspace;  // N * K
		zero_array_gpu(B, N * K);
		float* B0 = B;
		float* C = l->gpu.weight_grads;  // M * N
		for (int i = 0; i < (int)l->in_ids.n; i++) {
			layer* inl = l->in_layers[i];
			int c = (int)inl->out_c;
			float* im = &inl->gpu.output[s * (int)inl->out_n];
			im2col_gpu(im, B, w, h, c, out_w, out_h, ksize, stride, pad);
			B += N * l->ksize * l->ksize * c;
		}
		B = B0;
		gemm_atb_gpu(M, N, K, A, B, C, n_groups);
	}
	if (l->id == 0) return;
	for (int s = 0; s < batch_size; s++) {
		float* A = l->gpu.weights;  // M * N / n_groups
		float* B = &grads[s * M * K];  // M * K
		float* C = net->gpu.workspace;  // N * K
		zero_array_gpu(C, N * K);
		gemm_tab_gpu(M, N, K, A, B, C, n_groups);

		for (size_t i = 0; i < l->in_ids.n; i++) {
			layer* inl = l->in_layers[i];
			int c = (int)inl->out_c;
			float* im = &inl->gpu.grads[s * w * h * c];
			col2im_gpu(C, im, w, h, out_w, out_h, ksize, stride, pad, (int)l->n);
		}
	}
}

void update_conv_gpu(layer* l, network* net) {
	float rate = net->current_learning_rate;
	float momentum = net->momentum;
	size_t n_weights = l->n_weights;
	launch_update_kernel(l->gpu.biases, l->gpu.bias_grads, l->gpu.bias_velocities, (int)l->n_filters, momentum, rate);
	net->regularize_weights(l->gpu.weight_grads, l->gpu.weights, n_weights, net->decay);
	launch_update_kernel(l->gpu.weights, l->gpu.weight_grads, l->gpu.weight_velocities, (int)n_weights, momentum, rate);
	if (l->batchnorm) {
		launch_update_kernel(l->gpu.gammas, l->gpu.gamma_grads, l->gpu.gamma_velocities, (int)l->n_filters, momentum, rate);
	}
}
#else
#pragma warning (suppress:4100)
void forward_conv_gpu(layer* l, network* net) {
	gpu_not_defined();
}
#pragma warning (suppress:4100)
void backward_conv_gpu(layer* l, network* net) {
	gpu_not_defined();
}
#pragma warning (suppress:4100)
void update_conv_gpu(layer* l, network* net) {
	gpu_not_defined();
}
#endif

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
	l->n_weights = l->n_filters * l->ksize * l->ksize * l->c;
	l->weights = (float*)xcalloc(l->n_weights, sizeof(float));
	fill_array(l->weights, l->n_weights, 1.0F);
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

	net->workspace = (float*)xcalloc(l->out_w * l->out_h * l->ksize * l->ksize * l->c * 2, sizeof(float));

	forward_conv(l, net);

	pprint_mat(l->Z, (int)l->out_w, (int)l->out_h, (int)l->out_c);
}