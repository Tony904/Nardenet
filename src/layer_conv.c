#include "layer_conv.h"
#include <omp.h>
#include <assert.h>
#include <string.h>
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

#define BUFFSIZE 256

void forward_conv(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	size_t batch_size = net->batch_size;
	zero_array(l->Z, out.n * batch_size);
	size_t M = k.n_filters;
	size_t N = out.w * out.h;
	size_t K = k.ksize * k.ksize * in.c;
	float* A = l->weights;  // M * K
	for (size_t b = 0; b < batch_size; b++) {
		float* B = net->workspace;  // K * N
		float* B0 = B;
		for (size_t i = 0; i < l->n_inputs; i++) {
			layer_input input = l->inputs[i];
			float* im = &input.data[b * input.stride];
			im2col(im, B, (int)in.w, (int)in.h, (int)input.c, (int)out.w, (int)out.h, (int)k.ksize, (int)k.stride, (int)k.pad);
			B += k.ksize * k.ksize * input.c * N;
		}
		float* C = &l->Z[b * M * N];  // M * N
		gemm(M, N, K, A, B0, C, k.n_groups);
	}

	if (l->batchnorm) {
		forward_batchnorm(l, batch_size);
		l->activate(l->act_inputs, l->output, out.n, batch_size);
	}
	else {
		// note: l->Z = l->act_inputs when batchnorm disabled
		add_biases(l->Z, l->biases, M, N, batch_size);
		l->activate(l->Z, l->output, out.n, batch_size);
	}
	if (net->training) zero_array(l->grads, out.n * batch_size);
}

void backward_conv(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->grads;  // propogated gradients up to this layer
	// dz/dw = previous layer (shallower layer) input
	// da/dz = activation derivative
	get_activation_grads(l, batch_size); // dC/da * da/dz
	// grads[] is now dC/dz.
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	// now get dz/dw for each weight (it's just the input from the previous layer in the forward pass).
	size_t M = k.n_filters;
	size_t N = k.ksize * k.ksize * in.c; // filter length
	size_t K = out.w * out.h; // # of patches
	// sum dC/dz for each filter to get it's bias gradients.
	get_bias_grads(l->bias_grads, grads, M, K, batch_size);  // note: biases = betas for batch norm
	
	if (l->batchnorm) backward_batchnorm(l, batch_size);
	for (size_t s = 0; s < batch_size; s++) {
		float* A = &grads[s * M * K];  // M * K
		float* B = net->workspace;  // N * K
		zero_array(B, N * K);
		float* B0 = B;
		float* C = l->weight_grads;  // M * N
		for (size_t i = 0; i < l->n_inputs; i++) {
			layer_input input = l->inputs[i];
			float* im = &input.data[s * input.stride];
			im2col(im, B, (int)in.w, (int)in.h, (int)input.c, (int)out.w, (int)out.h, (int)k.ksize, (int)k.stride, (int)k.pad);
			//B += N * ksize * ksize * input.c;  // pretty sure this is wrong
			B += K * k.ksize * k.ksize * input.c;
		}
		B = B0;
		gemm_atb(M, N, K, A, B, C, k.n_groups);
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
		gemm_tab(M, N, K, A, B, C, k.n_groups);
		// C is now dC/da in col'd form (as in im2col).
		// So now we need to turn this "expanded" form (col) into the form of the dimensions of
		// the output of the input layer (im). We do this using col2im().
		for (size_t i = 0; i < l->n_inputs; i++) {
			layer_input input = l->inputs[i];
			float* im = &input.grads[s * input.stride];
			col2im(C, im, (int)in.w, (int)in.h, (int)input.c, (int)out.w, (int)out.h, (int)k.ksize, (int)k.stride, (int)k.pad);
			C += K * k.ksize * k.ksize * input.c;
		}
	}
}

void update_conv(layer* l, network* net) {
	float rate = net->current_learning_rate;	
	float momentum = net->momentum;
	float* biases = l->biases;
	float* bias_grads = l->bias_grads;
	float* bias_velocities = l->bias_velocities;
	size_t n = l->k.n_filters;
	size_t i;
#pragma omp parallel for
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
#pragma omp parallel for
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
		n = l->out.c;
#pragma omp parallel for
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
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	int batch_size = (int)net->batch_size;
	zero_array_gpu(l->gpu.Z, (int)(out.n * batch_size));
	int M = (int)k.n_filters;
	int N = (int)(out.w * out.h);
	int K = (int)(k.ksize * k.ksize * in.c);
	float* A = l->gpu.weights;  // M * K
	for (int b = 0; b < batch_size; b++) {
		float* B = net->gpu.workspace;  // K * N
		float* B0 = B;
		for (int i = 0; i < (int)l->n_inputs; i++) {
			layer_input input = l->inputs[i];
			float* im = &input.gpu.data[b * input.stride];
			im2col_gpu(im, B, (int)in.w, (int)in.h, (int)input.c, (int)out.w, (int)out.h, (int)k.ksize, (int)k.stride, (int)k.pad);
			B += N * k.ksize * k.ksize * input.c;
		}
		float* C = &l->gpu.Z[b * M * N];  // M * N
		gemm_gpu(M, N, K, A, B0, C, (int)k.n_groups);
	}
	if (l->batchnorm) {
		forward_batchnorm_gpu(l->gpu.gammas, l->gpu.biases, l->gpu.means, l->gpu.variances, l->gpu.rolling_means, l->gpu.rolling_variances, l->gpu.Z, l->gpu.Z_norm, l->gpu.act_inputs, N, M, batch_size);
		l->activate(l->gpu.act_inputs, l->gpu.output, out.n, batch_size);
	}
	else {
		add_biases_gpu(l->gpu.Z, l->gpu.biases, M, N, batch_size);
		l->activate(l->gpu.Z, l->gpu.output, out.n, batch_size);
	}
	if (net->training) zero_array_gpu(l->gpu.grads, (int)(out.n * batch_size));
}

void backward_conv_gpu(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	int batch_size = (int)net->batch_size;
	float* grads = l->gpu.grads;

	get_activation_grads_gpu(l, batch_size);
	int M = (int)k.n_filters;
	int N = (int)(k.ksize * k.ksize * in.c); // weights per filter
	int K = (int)(out.w * out.h); // # of patches

	get_bias_grads_gpu(l->gpu.bias_grads, grads, M, K, batch_size);

	if (l->batchnorm) backward_batchnorm_gpu(grads, l->gpu.Z, l->gpu.Z_norm, l->gpu.means, l->gpu.variances, l->gpu.gammas, l->gpu.gamma_grads, K, M, batch_size);

	for (int s = 0; s < batch_size; s++) {
		float* A = &grads[s * M * K];  // M * K
		float* B = net->gpu.workspace;  // N * K
		zero_array_gpu(B, N * K);
		float* B0 = B;
		for (int i = 0; i < (int)l->n_inputs; i++) {
			layer_input input = l->inputs[i];
			float* im = &input.gpu.data[s * input.stride];
			im2col_gpu(im, B, (int)in.w, (int)in.h, (int)input.c, (int)out.w, (int)out.h, (int)k.ksize, (int)k.stride, (int)k.pad);
			B += K * l->k.ksize * l->k.ksize * input.c;
		}
		float* C = l->gpu.weight_grads;  // M * N
		B = B0;
		gemm_atb_gpu(M, N, K, A, B, C, (int)k.n_groups);
	}
	if (l->id == 0) return;
	for (int s = 0; s < batch_size; s++) {
		float* A = l->gpu.weights;  // M * N / n_groups
		float* B = &grads[s * M * K];  // M * K
		float* C = net->gpu.workspace;  // N * K
		zero_array_gpu(C, N * K);
		gemm_tab_gpu(N, K, M, A, B, C, (int)k.n_groups);

		for (size_t i = 0; i < l->n_inputs; i++) {
			layer_input input = l->inputs[i];
			float* im = &input.gpu.grads[s * input.stride];
			col2im_gpu(C, im, (int)in.w, (int)in.h, (int)out.w, (int)out.h, (int)k.ksize, (int)k.stride, (int)k.pad, (int)(in.w * in.h * input.c));
			C += K * k.ksize * k.ksize * input.c;
		}
	}
}

void update_conv_gpu(layer* l, network* net) {
	float rate = net->current_learning_rate;
	float momentum = net->momentum;
	size_t n_weights = l->n_weights;
	launch_update_kernel(l->gpu.biases, l->gpu.bias_grads, l->gpu.bias_velocities, (int)l->k.n_filters, momentum, rate);
	net->regularize_weights(l->gpu.weight_grads, l->gpu.weights, n_weights, net->decay);
	launch_update_kernel(l->gpu.weights, l->gpu.weight_grads, l->gpu.weight_velocities, (int)n_weights, momentum, rate);
	if (l->batchnorm) {
		launch_update_kernel(l->gpu.gammas, l->gpu.gamma_grads, l->gpu.gamma_velocities, (int)l->k.n_filters, momentum, rate);
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


void forward_conv_cpu_gpu_compare(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	char buff[BUFFSIZE] = { 0 };
	size_t batch_size = net->batch_size;
	zero_array(l->Z, out.n * batch_size);
	zero_array_gpu(l->gpu.Z, (int)(out.n * batch_size));
	compare_cpu_gpu_arrays(l->Z, l->gpu.Z, out.n * batch_size, l->id, "forward conv, Z, zero_array");

	size_t n_groups = k.n_groups;
	size_t M = k.n_filters;
	size_t N = out.w * out.h;
	size_t K = k.ksize * k.ksize * in.c;
	float* A_cpu = l->weights;  // M * K
	float* A_gpu = l->gpu.weights;
	compare_cpu_gpu_arrays(l->weights, l->gpu.weights, l->n_weights, l->id, "forward conv, weights, pre-im2col");

	for (size_t b = 0; b < batch_size; b++) {
		float* B_cpu = net->workspace;  // K * N
		float* B_gpu = net->gpu.workspace;
		for (size_t i = 0; i < l->n_inputs; i++) {
			layer_input input = l->inputs[i];
			float* im_cpu = &input.data[b * input.stride];
			float* im_gpu = &input.gpu.data[b * input.stride];

			memset(buff, 0, sizeof(buff));
			snprintf(buff, BUFFSIZE, "%s%zu", "forward conv, im array, pre-im2col, b=", b);
			compare_cpu_gpu_arrays(im_cpu, im_gpu, in.w * in.h * input.c, l->id, buff);

			im2col(im_cpu, B_cpu, (int)in.w, (int)in.h, (int)input.c, (int)out.w, (int)out.h, (int)k.ksize, (int)k.stride, (int)k.pad);
			im2col_gpu(im_gpu, B_gpu, (int)in.w, (int)in.h, (int)input.c, (int)out.w, (int)out.h, (int)k.ksize, (int)k.stride, (int)k.pad);

			memset(buff, 0, sizeof(buff));
			snprintf(buff, BUFFSIZE, "%s%zu", "forward conv, B array (col), post-im2col, b=", b);
			compare_cpu_gpu_arrays(B_cpu, B_gpu, K * N, l->id, buff);

			B_cpu += N * k.ksize * k.ksize * input.c;
			B_gpu += N * k.ksize * k.ksize * input.c;
		}
		float* C_cpu = &l->Z[b * M * N];  // M * N
		float* C_gpu = &l->gpu.Z[b * M * N];

		memset(buff, 0, sizeof(buff));
		snprintf(buff, BUFFSIZE, "%s%zu", "forward conv, C array (Z), pre-gemm, b=", b);
		compare_cpu_gpu_arrays(C_cpu, C_gpu, M * N, l->id, buff);

		gemm(M, N, K, A_cpu, net->workspace, C_cpu, n_groups);
		gemm_gpu((int)M, (int)N, (int)K, A_gpu, net->gpu.workspace, C_gpu, (int)n_groups);

		memset(buff, 0, sizeof(buff));
		snprintf(buff, BUFFSIZE, "%s%zu", "forward conv, C array (Z), post-gemm, b=", b);
		compare_cpu_gpu_arrays(C_cpu, C_gpu, M * N, l->id, buff);
	}
	if (l->batchnorm) {
		compare_cpu_gpu_arrays(l->gammas, l->gpu.gammas, k.n_filters, l->id, "forward conv, gammas, pre-batchnorm");
		compare_cpu_gpu_arrays(l->biases, l->gpu.biases, k.n_filters, l->id, "forward conv, biases, pre-batchnorm");
		compare_cpu_gpu_arrays(l->means, l->gpu.means, k.n_filters, l->id, "forward conv, means, pre-batchnorm");
		compare_cpu_gpu_arrays(l->variances, l->gpu.variances, k.n_filters, l->id, "forward conv, variances, pre-batchnorm");
		compare_cpu_gpu_arrays(l->rolling_means, l->gpu.rolling_means, k.n_filters, l->id, "forward conv, rolling means, pre-batchnorm");
		compare_cpu_gpu_arrays(l->rolling_variances, l->gpu.rolling_variances, k.n_filters, l->id, "forward conv, rolling variances, pre-batchnorm");
		compare_cpu_gpu_arrays(l->Z, l->gpu.Z, k.n_filters, l->id, "forward conv, Z, pre-batchnorm");
		compare_cpu_gpu_arrays(l->Z_norm, l->gpu.Z_norm, k.n_filters, l->id, "forward conv, Z_norm, pre-batchnorm");
		compare_cpu_gpu_arrays(l->act_inputs, l->gpu.act_inputs, k.n_filters, l->id, "forward, act_inputs, pre-batchnorm");

		forward_batchnorm(l, batch_size);
		forward_batchnorm_gpu(l->gpu.gammas, l->gpu.biases, l->gpu.means, l->gpu.variances, l->gpu.rolling_means, l->gpu.rolling_variances, l->gpu.Z, l->gpu.Z_norm, l->gpu.act_inputs, (int)N, (int)M, (int)batch_size);

		compare_cpu_gpu_arrays(l->gammas, l->gpu.gammas, k.n_filters, l->id, "forward conv, gammas, post-batchnorm");
		compare_cpu_gpu_arrays(l->biases, l->gpu.biases, k.n_filters, l->id, "forward conv, biases, post-batchnorm");
		compare_cpu_gpu_arrays(l->means, l->gpu.means, k.n_filters, l->id, "forward conv, means, post-batchnorm");
		compare_cpu_gpu_arrays(l->variances, l->gpu.variances, k.n_filters, l->id, "forward conv, variances, post-batchnorm");
		compare_cpu_gpu_arrays(l->rolling_means, l->gpu.rolling_means, k.n_filters, l->id, "forward conv, rolling means, post-batchnorm");
		compare_cpu_gpu_arrays(l->rolling_variances, l->gpu.rolling_variances, k.n_filters, l->id, "forward conv, rolling variances, post-batchnorm");
		compare_cpu_gpu_arrays(l->Z, l->gpu.Z, k.n_filters, l->id, "forward conv, Z, post-batchnorm");
		compare_cpu_gpu_arrays(l->Z_norm, l->gpu.Z_norm, k.n_filters, l->id, "forward conv, Z_norm, post-batchnorm");
		compare_cpu_gpu_arrays(l->act_inputs, l->gpu.act_inputs, k.n_filters, l->id, "forward, act_inputs, post-batchnorm");

		activate(l->act_inputs, l->output, out.n, batch_size, l->activation, 0);
		activate(l->gpu.act_inputs, l->gpu.output, out.n, batch_size, l->activation, 1);

		compare_cpu_gpu_arrays(l->output, l->gpu.output, out.n * batch_size, l->id, "forward conv, output, post-activation");
	}
	else {
		// note: l->Z = l->act_inputs when batchnorm disabled
		add_biases(l->Z, l->biases, M, N, batch_size);
		add_biases_gpu(l->gpu.Z, l->gpu.biases, (int)M, (int)N, (int)batch_size);

		compare_cpu_gpu_arrays(l->Z, l->gpu.Z, M * N, l->id, "forward conv, Z, after adding biases");

		activate(l->Z, l->output, out.n, batch_size, l->activation, 0);
		activate(l->gpu.Z, l->gpu.output, out.n, batch_size, l->activation, 1);

		compare_cpu_gpu_arrays(l->output, l->gpu.output, out.n * batch_size, l->id, "forward conv, output, post-activation");
	}
	if (net->training) {
		zero_array(l->grads, out.n * batch_size);
		zero_array_gpu(l->gpu.grads, (int)(out.n * batch_size));

		compare_cpu_gpu_arrays(l->grads, l->gpu.grads, out.n * batch_size, l->id, "forward conv, grads, after zeroing");
	}
}

void backward_conv_cpu_gpu_compare(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	char buff[BUFFSIZE] = { 0 };

	size_t batch_size = net->batch_size;
	float* grads_cpu = l->grads;
	float* grads_gpu = l->gpu.grads;

	compare_cpu_gpu_arrays(grads_cpu, grads_gpu, batch_size *  out.n, l->id, "backward conv, grads, pre-activation grads");

	get_activation_grads(l, batch_size);
	get_activation_grads_gpu(l, batch_size);

	compare_cpu_gpu_arrays(grads_cpu, grads_gpu, batch_size *  out.n, l->id, "backward conv, grads, post-activation grads");

	size_t M = k.n_filters;
	size_t N = k.ksize * k.ksize * in.c;
	size_t K = out.w * out.h;

	compare_cpu_gpu_arrays(l->bias_grads, l->gpu.bias_grads, k.n_filters, l->id, "backward conv, bias grads, pre-get bias grads");

	get_bias_grads(l->bias_grads, grads_cpu, M, K, batch_size);  // note: biases = betas for batch norm
	get_bias_grads_gpu(l->gpu.bias_grads, grads_gpu, (int)M, (int)K, (int)batch_size);

	compare_cpu_gpu_arrays(l->bias_grads, l->gpu.bias_grads, k.n_filters, l->id, "backward conv, bias grads, post-get bias grads");

	if (l->batchnorm) {
		compare_cpu_gpu_arrays(l->gammas, l->gpu.gammas, k.n_filters, l->id, "backward conv, gammas, pre-batchnorm");
		compare_cpu_gpu_arrays(l->means, l->gpu.means, k.n_filters, l->id, "backward conv, means, pre-batchnorm");
		compare_cpu_gpu_arrays(l->variances, l->gpu.variances, k.n_filters, l->id, "backward conv, variances, pre-batchnorm");
		compare_cpu_gpu_arrays(l->Z, l->gpu.Z, k.n_filters * batch_size, l->id, "backward conv, Z, pre-batchnorm");
		compare_cpu_gpu_arrays(l->Z_norm, l->gpu.Z_norm, k.n_filters * batch_size, l->id, "backward conv, Z_norm, pre-batchnorm");
		compare_cpu_gpu_arrays(l->gamma_grads, l->gpu.gamma_grads, k.n_filters, l->id, "backward conv, gamma grads, pre-batchnorm");

		backward_batchnorm(l, batch_size);
		backward_batchnorm_gpu(grads_gpu, l->gpu.Z, l->gpu.Z_norm, l->gpu.means, l->gpu.variances, l->gpu.gammas, l->gpu.gamma_grads, (int)K, (int)M, (int)batch_size);

		compare_cpu_gpu_arrays(l->grads, l->gpu.grads,  out.n * batch_size, l->id, "backward conv, grads, post_batchnorm");
		compare_cpu_gpu_arrays(l->gammas, l->gpu.gammas, k.n_filters, l->id, "backward conv, gammas, post-batchnorm");
		compare_cpu_gpu_arrays(l->means, l->gpu.means, k.n_filters, l->id, "backward conv, means, post-batchnorm");
		compare_cpu_gpu_arrays(l->variances, l->gpu.variances, k.n_filters, l->id, "backward conv, variances, post-batchnorm");
		compare_cpu_gpu_arrays(l->Z, l->gpu.Z, k.n_filters * batch_size, l->id, "backward conv, Z, post-batchnorm");
		compare_cpu_gpu_arrays(l->Z_norm, l->gpu.Z_norm, k.n_filters * batch_size, l->id, "backward conv, Z_norm, post-batchnorm");
		compare_cpu_gpu_arrays(l->gamma_grads, l->gpu.gamma_grads, k.n_filters, l->id, "backward conv, gamma grads, post-batchnorm");
	}
	for (size_t b = 0; b < batch_size; b++) {
		float* A_cpu = &grads_cpu[b * M * K];  // M * K
		float* A_gpu = &grads_gpu[b * M * K];

		memset(buff, 0, sizeof(buff));
		snprintf(buff, BUFFSIZE, "%s%zu", "backward conv, A array, pre-im2col, b=", b);
		compare_cpu_gpu_arrays(A_cpu, A_gpu, M * K, l->id, buff);

		float* B_cpu = net->workspace;  // N * K
		float* B_gpu = net->gpu.workspace;
		zero_array(B_cpu, N * K);
		zero_array_gpu(B_gpu, (int)(N * K));

		memset(buff, 0, sizeof(buff));
		snprintf(buff, BUFFSIZE, "%s%zu", "backward conv, B array (col), ater zeroing, pre-im2col, b=", b);
		compare_cpu_gpu_arrays(B_cpu, B_gpu, N * K, l->id, buff);

		for (size_t i = 0; i < l->n_inputs; i++) {
			layer_input input = l->inputs[i];
			float* im_cpu = &input.data[b * input.stride];
			float* im_gpu = &input.gpu.data[b * input.stride];

			memset(buff, 0, sizeof(buff));
			snprintf(buff, BUFFSIZE, "%s%zu", "backward conv, im array, pre-im2col, b=", b);
			compare_cpu_gpu_arrays(im_cpu, im_gpu, (int)in.n, l->id, buff);

			im2col(im_cpu, B_cpu, (int)in.w, (int)in.h, (int)input.c, (int)out.w, (int)out.h, (int)k.ksize, (int)k.stride, (int)k.pad);
			im2col_gpu(im_gpu, B_gpu, (int)in.w, (int)in.h, (int)input.c, (int)out.w, (int)out.h, (int)k.ksize, (int)k.stride, (int)k.pad);

			memset(buff, 0, sizeof(buff));
			snprintf(buff, BUFFSIZE, "%s%zu", "backward conv, B array (col), post-im2col, b=", b);
			compare_cpu_gpu_arrays(B_cpu, B_gpu, K * N, l->id, buff);

			B_cpu += K * k.ksize * k.ksize * input.c;
			B_gpu += K * k.ksize * k.ksize * input.c;
		}
		float* C_cpu = l->weight_grads;  // M * N
		float* C_gpu = l->gpu.weight_grads;

		memset(buff, 0, sizeof(buff));
		snprintf(buff, BUFFSIZE, "%s%zu", "backward conv, C array (w grads), pre-gemm atb, b=", b);
		compare_cpu_gpu_arrays(C_cpu, C_gpu, M * N, l->id, buff);

		gemm_atb(M, N, K, A_cpu, net->workspace, C_cpu, k.n_groups);
		gemm_atb_gpu((int)M, (int)N, (int)K, A_gpu, net->gpu.workspace, C_gpu, (int)k.n_groups);

		memset(buff, 0, sizeof(buff));
		snprintf(buff, BUFFSIZE, "%s%zu", "backward conv, C array (w grads), post-gemm atb, b=", b);
		compare_cpu_gpu_arrays(C_cpu, C_gpu, M * N, l->id, buff);
	}
	if (l->id == 0) return;
	for (size_t b = 0; b < batch_size; b++) {
		float* A_cpu = l->weights;  // M * N / n_groups
		float* A_gpu = l->gpu.weights;
		float* B_cpu = &grads_cpu[b * M * K];  // M * K
		float* B_gpu = &grads_gpu[b * M * K];
		float* C_cpu = net->workspace;  // N * K
		float* C_gpu = net->gpu.workspace;
		zero_array(C_cpu, N * K);
		zero_array_gpu(C_gpu, (int)(N * K));

		memset(buff, 0, sizeof(buff));
		snprintf(buff, BUFFSIZE, "%s%zu", "backward conv, C array (workspace), after zeroing, pre-gemm tab, b=", b);
		compare_cpu_gpu_arrays(C_cpu, C_gpu, K * N, l->id, buff);

		gemm_tab(M, N, K, A_cpu, B_cpu, C_cpu, k.n_groups);
		gemm_tab_gpu((int)N, (int)K, (int)M, A_gpu, B_gpu, C_gpu, (int)k.n_groups);

		memset(buff, 0, sizeof(buff));
		snprintf(buff, BUFFSIZE, "%s%zu", "backward conv, C array (workspace), post-gemm tab, b=", b);
		compare_cpu_gpu_arrays(C_cpu, C_gpu, K * N, l->id, buff);

		for (size_t i = 0; i < l->n_inputs; i++) {
			layer_input input = l->inputs[i];
			float* im_cpu = &input.grads[b * input.stride];
			float* im_gpu = &input.gpu.grads[b * input.stride];

			memset(buff, 0, sizeof(buff));
			snprintf(buff, BUFFSIZE, "%s%zu", "backward conv, im array, pre-col2im, b=", b);
			compare_cpu_gpu_arrays(im_cpu, im_gpu, in.w * in.h * input.c, l->id, buff);

			col2im(C_cpu, im_cpu, (int)in.w, (int)in.h, (int)input.c, (int)out.w, (int)out.h, (int)k.ksize, (int)k.stride, (int)k.pad);
			col2im_gpu(C_gpu, im_gpu, (int)in.w, (int)in.h, (int)out.w, (int)out.h, (int)k.ksize, (int)k.stride, (int)k.pad, (int)input.n);

			memset(buff, 0, sizeof(buff));
			snprintf(buff, BUFFSIZE, "%s%zu", "backward conv, im array, post-col2im, b=", b);
			compare_cpu_gpu_arrays(im_cpu, im_gpu, (int)input.n, l->id, buff);

			C_cpu += K * k.ksize * k.ksize * input.c;
			C_gpu += K * k.ksize * k.ksize * input.c;
		}
	}
}

void update_conv_cpu_gpu_compare(layer* l, network* net) {
	float rate = net->current_learning_rate;
	float momentum = net->momentum;

	float* biases = l->biases;
	float* bias_grads = l->bias_grads;
	float* bias_velocities = l->bias_velocities;
	kernel_params k = l->k;
	size_t n = k.n_filters;

	compare_cpu_gpu_arrays(l->biases, l->gpu.biases, k.n_filters, l->id, "update conv, biases, pre-update");
	compare_cpu_gpu_arrays(l->bias_velocities, l->gpu.bias_velocities, k.n_filters, l->id, "update conv, bias velocities, pre-update");
	compare_cpu_gpu_arrays(l->bias_grads, l->gpu.bias_grads, k.n_filters, l->id, "update conv, bias grads, pre-update");

	size_t i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		float v_old = bias_velocities[i];
		float v_new = momentum * v_old - rate * bias_grads[i];
		biases[i] += -momentum * v_old + (1.0F + momentum) * v_new;  // Nesterov momentum
		bias_velocities[i] = v_new;
		bias_grads[i] = 0.0F;
	}
	launch_update_kernel(l->gpu.biases, l->gpu.bias_grads, l->gpu.bias_velocities, (int)k.n_filters, momentum, rate);

	compare_cpu_gpu_arrays(l->biases, l->gpu.biases, k.n_filters, l->id, "update conv, biases, post-update");
	compare_cpu_gpu_arrays(l->bias_velocities, l->gpu.bias_velocities, k.n_filters, l->id, "update conv, bias velocities, post-update");
	compare_cpu_gpu_arrays(l->bias_grads, l->gpu.bias_grads, k.n_filters, l->id, "update conv, bias grads, post-update");

	net->regularize_weights(l->weight_grads, l->weights, l->n_weights, net->decay);
	net->regularize_weights(l->gpu.weight_grads, l->gpu.weights, l->n_weights, net->decay);

	compare_cpu_gpu_arrays(l->weights, l->gpu.weights, l->n_weights, l->id, "update conv, weights, pre-update");
	compare_cpu_gpu_arrays(l->weight_grads, l->gpu.weight_grads, l->n_weights, l->id, "update conv, weight grads, pre-update");
	compare_cpu_gpu_arrays(l->weight_velocities, l->gpu.weight_velocities, l->n_weights, l->id, "update conv, weight velocities, pre-update");

	float* weights = l->weights;
	float* weight_grads = l->weight_grads;
	float* weight_velocities = l->weight_velocities;
	n = l->n_weights;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		float v_old = weight_velocities[i];
		float v_new = momentum * v_old - rate * weight_grads[i];
		weights[i] += -momentum * v_old + (1.0F + momentum) * v_new;
		weight_velocities[i] = v_new;
		weight_grads[i] = 0.0F;
	}
	launch_update_kernel(l->gpu.weights, l->gpu.weight_grads, l->gpu.weight_velocities, (int)l->n_weights, momentum, rate);

	compare_cpu_gpu_arrays(l->weights, l->gpu.weights, l->n_weights, l->id, "update conv, weights, post-update");
	compare_cpu_gpu_arrays(l->weight_grads, l->gpu.weight_grads, l->n_weights, l->id, "update conv, weight grads, post-update");
	compare_cpu_gpu_arrays(l->weight_velocities, l->gpu.weight_velocities, l->n_weights, l->id, "update conv, weight velocities, post-update");

	if (l->batchnorm) {
		float* gammas = l->gammas;
		float* gamma_grads = l->gamma_grads;
		float* gamma_velocities = l->gamma_velocities;
		n = l->out.c;
#pragma omp parallel for
		for (i = 0; i < n; i++) {
			float v_old = weight_velocities[i];
			float v_new = momentum * v_old - rate * weight_grads[i];
			gammas[i] += -momentum * v_old + (1.0F + momentum) * v_new;
			gamma_velocities[i] = v_new;
			gamma_grads[i] = 0.0F;
		}
		launch_update_kernel(l->gpu.gammas, l->gpu.gamma_grads, l->gpu.gamma_velocities, (int)k.n_filters, momentum, rate);
	}
}

/*** TESTS ***/

//void test_forward_conv(void) {
//	layer* l = (layer*)xcalloc(1, sizeof(layer));
//	network* net = (network*)xcalloc(1, sizeof(network));
//	l->n_filters = 2;
//	k.ksize = 2;
//	l->pad = 0;
//	l->stride = 1;
//	l->w = 3;
//	l->h = 3;
//	l->c = 3;
//	out.w = (l->w + 2 * l->pad - k.ksize) / l->stride + 1;
//	out.h = (l->h + 2 * l->pad - k.ksize) / l->stride + 1;
//	out.c = l->n_filters;
//	l->biases = (float*)xcalloc(l->n_filters, sizeof(float));
//	fill_array(l->biases, l->n_filters, 0.5F);
//	l->activate = activate_none;
//	l->n_weights = l->n_filters * k.ksize * k.ksize * l->c;
//	l->weights = (float*)xcalloc(l->n_weights, sizeof(float));
//	fill_array(l->weights, l->n_weights, 1.0F);
//	l->Z = (float*)xcalloc(l->n_filters * out.w * out.h, sizeof(float));
//	layer_input input1 = (layer*)xcalloc(1, sizeof(layer));
//	layer_input input2 = (layer*)xcalloc(1, sizeof(layer));
//	inl1->out_w = l->w;
//	inl2->out_w = l->w;
//	inl1->out_h = l->h;
//	inl2->out_h = l->h;
//	inl1->out_c = 1;
//	inl2->out_c = 2;
//	assert(l->c == inl1->out_c + inl2->out_c);
//	inl1->out_n = inl1->out_w * inl1->out_h * inl1->out_c;
//	inl2->out_n = inl2->out_w * inl2->out_h * inl2->out_c;
//	inl1->output = (float*)xcalloc(inl1->out_n, sizeof(float));
//	fill_array(inl1->output, inl1->out_n, 2.0F);
//	inl2->output = (float*)xcalloc(inl2->out_n, sizeof(float));
//	fill_array(inl2->output, inl2->out_n, 1.0F);
//	l->in_layers = (layer**)xcalloc(2, sizeof(layer*));
//	l->inputs[0] = inl1;
//	l->inputs[1] = inl2;
//	l->n_inputs = 2;
//
//	net->workspace = (float*)xcalloc(out.w * out.h * k.ksize * k.ksize * l->c * 2, sizeof(float));
//
//	forward_conv(l, net);
//
//	pprint_mat(l->Z, (int)out.w, (int)out.h, (int)out.c);
//}