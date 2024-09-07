#include "layer_residual.h"
#include "derivatives.h"
#include "utils.h"
#include "xallocs.h"



void forward_residual(layer* l, network* net) {
	size_t n = l->out_n * net->batch_size;
	float* l_Z = l->Z;
	float* inl0_output = l->in_layers[0]->output;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		l_Z[i] = inl0_output[i];
	}
	for (size_t a = 1; a < l->in_ids.n; a++) {
		float* inl_output = l->in_layers[a]->output;
#pragma omp parallel for
		for (i = 0; i < n; i++) {
			l_Z[i] += inl_output[i];
		}
	}
	if (l->activation) l->activate(l_Z, l->output, l->out_n, net->batch_size);
	if (net->training) zero_array(l->grads, n);
}

void backward_residual(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->grads;
	if (l->activation) {
		if (l->activation == ACT_MISH) get_grads_mish(grads, l->act_inputs, l->out_n, batch_size);  // dC/da * da/dz
		else if (l->activation == ACT_RELU) get_grads_relu(grads, l->act_inputs, l->out_n, batch_size);
		else if (l->activation == ACT_LEAKY) get_grads_leaky_relu(grads, l->act_inputs, l->out_n, batch_size);
		else if (l->activation == ACT_SIGMOID) get_grads_sigmoid(grads, l->output, l->out_n, batch_size);
		else if (l->activation == ACT_TANH) get_grads_tanh(grads, l->act_inputs, l->out_n, batch_size);
		else {
			printf("Incorrect or unsupported activation function.\n");
			wait_for_key_then_exit();
		}
	}
	size_t N = batch_size * l->out_n;
	for (size_t a = 0; a < l->in_ids.n; a++) {
		float* inl_grads = l->in_layers[a]->grads;
		size_t i;
#pragma omp parallel for
		for (i = 0; i < N; i++) {
			inl_grads[i] += grads[i];
		}
	}
}

/*** TESTING ***/

void test_forward_residual(void) {
	layer l = { 0 };
	network net = { 0 };
	net.batch_size = 2;
	l.w = 4;
	l.h = 4;
	l.c = 2;
	l.n = l.w * l.h * l.c;
	l.out_w = l.w;
	l.out_h = l.h;
	l.out_c = l.c;
	l.out_n = l.out_w * l.out_h * l.out_c;
	l.output = (float*)xcalloc(l.out_n * net.batch_size, sizeof(float));
	l.Z = l.output;
	layer inl1 = { 0 };
	layer inl2 = { 0 };
	inl1.out_w = l.w;
	inl2.out_w = l.w;
	inl1.out_h = l.h;
	inl2.out_h = l.h;
	inl1.out_c = l.c;
	inl2.out_c = l.c;
	inl1.out_n = inl1.out_w * inl1.out_h * inl1.out_c;
	inl2.out_n = inl2.out_w * inl2.out_h * inl2.out_c;
	inl1.output = (float*)xcalloc(inl1.out_n * net.batch_size, sizeof(float));
	fill_array_rand_float(inl1.output, inl1.out_n * net.batch_size, 0.0, 1.0);
	inl2.output = (float*)xcalloc(inl2.out_n * net.batch_size, sizeof(float));
	fill_array_rand_float(inl2.output, inl2.out_n * net.batch_size, 0.0, 1.0);
	inl1.grads = (float*)xcalloc(inl1.out_n * net.batch_size, sizeof(float));
	inl2.grads = (float*)xcalloc(inl2.out_n * net.batch_size, sizeof(float));
	l.in_layers = (layer**)xcalloc(2, sizeof(layer*));
	l.in_layers[0] = &inl1;
	l.in_layers[1] = &inl2;
	l.in_ids.n = 2;
	forward_residual(&l, &net);

	pprint_mat(inl1.output, (int)inl1.out_w, (int)inl1.out_h, (int)inl1.out_c * (int)net.batch_size);
	pprint_mat(inl2.output, (int)inl2.out_w, (int)inl2.out_h, (int)inl2.out_c * (int)net.batch_size);
	pprint_mat(l.output, (int)l.out_w, (int)l.out_h, (int)l.out_c * (int)net.batch_size);
}

void test_backward_residual(void) {
	layer l = { 0 };
	network net = { 0 };
	net.batch_size = 2;
	l.w = 4;
	l.h = 4;
	l.c = 2;
	l.n = l.w * l.h * l.c;
	l.out_w = l.w;
	l.out_h = l.h;
	l.out_c = l.c;
	l.out_n = l.out_w * l.out_h * l.out_c;
	l.output = (float*)xcalloc(l.out_n * net.batch_size, sizeof(float));
	l.grads = (float*)xcalloc(l.out_n * net.batch_size, sizeof(float));
	l.Z = l.output;
	layer inl1 = { 0 };
	layer inl2 = { 0 };
	inl1.out_w = l.w;
	inl2.out_w = l.w;
	inl1.out_h = l.h;
	inl2.out_h = l.h;
	inl1.out_c = l.c;
	inl2.out_c = l.c;
	inl1.out_n = inl1.out_w * inl1.out_h * inl1.out_c;
	inl2.out_n = inl2.out_w * inl2.out_h * inl2.out_c;
	inl1.output = (float*)xcalloc(inl1.out_n * net.batch_size, sizeof(float));
	fill_array_rand_float(inl1.output, inl1.out_n * net.batch_size, 0.0, 1.0);
	inl2.output = (float*)xcalloc(inl2.out_n * net.batch_size, sizeof(float));
	fill_array_rand_float(inl2.output, inl2.out_n * net.batch_size, 0.0, 1.0);
	inl1.grads = (float*)xcalloc(inl1.out_n * net.batch_size, sizeof(float));
	inl2.grads = (float*)xcalloc(inl2.out_n * net.batch_size, sizeof(float));
	l.in_layers = (layer**)xcalloc(2, sizeof(layer*));
	l.in_layers[0] = &inl1;
	l.in_layers[1] = &inl2;
	l.in_ids.n = 2;
	forward_residual(&l, &net);

	// note: in_layers are not concatinated for residual layers, only added.
	fill_array_rand_float(l.grads, l.out_n * net.batch_size, 5.0, 1.0);
	backward_residual(&l, &net);
	pprint_mat(inl1.grads, (int)inl1.out_w, (int)inl1.out_h, (int)(inl1.out_c * net.batch_size));
	pprint_mat(inl2.grads, (int)inl2.out_w, (int)inl2.out_h, (int)(inl2.out_c * net.batch_size));
}