#include "layer_avgpool.h"
#include <omp.h>
#include "derivatives.h"
#include "utils.h"
#include "xallocs.h"
#include "xcuda.h"
#include "blas.h"


void forward_avgpool(layer* l, network* net) {
	float* Z = l->Z;
	size_t w = l->w;
	size_t h = l->h;
	size_t wh = w * h;
	size_t batch_size = net->batch_size;
	for (size_t b = 0; b < batch_size; b++) {
		size_t bwh = b * wh;
		for (size_t a = 0; a < l->in_ids.n; a++) {
			layer* inl = l->in_layers[a];
			float* inl_output = inl->output;
			size_t inl_out_c = inl->out_c;
			size_t b_offset = bwh * inl_out_c;
			size_t ch;
#pragma omp parallel for firstprivate(wh)
			for (ch = 0; ch < inl_out_c; ch++) {
				float sum = 0.0F;
				size_t offset = ch * wh + b_offset;
				for (size_t s = 0; s < wh; s++) {
					sum += inl_output[offset + s];
				}
				Z[ch] = sum / wh;
			}
			Z += inl_out_c;
		}
	}
	if (l->activation) l->activate(l->Z, l->output, l->out_n, net->batch_size);
	if (net->training) zero_array(l->grads, l->out_n * batch_size);
}

void backward_avgpool(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->grads;
	if (l->activation) get_activation_grads(l, batch_size);
	size_t w = l->w;
	size_t h = l->h;
	size_t wh = w * h;
	float whF = (float)wh;
	for (size_t b = 0; b < batch_size; b++) {
		size_t bwh = b * wh;
		for (size_t a = 0; a < l->in_ids.n; a++) {
			layer* inl = l->in_layers[a];
			float* inl_grads = inl->grads;
			size_t inl_out_c = inl->out_c;
			size_t b_offset = bwh * inl_out_c;
			size_t ch;
#pragma omp parallel for firstprivate(wh, b_offset, whF)
			for (ch = 0; ch < inl_out_c; ch++) {
				size_t offset = ch * wh + b_offset;
				float grad = grads[ch] / whF;
				for (size_t s = 0; s < wh; s++) {
					inl_grads[offset + s] += grad;
				}
			}
			grads += inl_out_c;
		}
	}
}

void forward_avgpool_gpu(layer* l, network* net) {
	float* Z = l->Z;
	size_t w = l->w;
	size_t h = l->h;
	size_t wh = w * h;
	size_t batch_size = net->batch_size;
	if (l->in_ids.n > 1) {
		for (size_t b = 0; b < batch_size; b++) {
			size_t bwh = b * wh;
			for (size_t a = 0; a < l->in_ids.n; a++) {
				layer* inl = l->in_layers[a];
				float* inl_output = inl->output;
				size_t inl_out_c = inl->out_c;
				size_t b_offset = bwh * inl_out_c;
				launch_forward_avgpool_kernel(&inl_output[b_offset], Z, (int)wh, (int)inl_out_c, 1);
				Z += inl_out_c;
			}
		}
	}
	else launch_forward_avgpool_kernel(l->in_layers[0]->output, Z, (int)wh, (int)l->c, (int)batch_size);
	
	if (l->activation) l->activate(l->Z, l->output, l->out_n, net->batch_size);
	if (net->training) zero_array_gpu(l->grads, (int)(l->out_n * batch_size));
}

void backward_avgpool_gpu(layer* l, network* net) {
	int batch_size = (int)net->batch_size;
	float* grads = l->grads;
	if (l->activation) get_activation_grads_gpu(l, batch_size);
	size_t w = l->w;
	size_t h = l->h;
	size_t wh = w * h;
	if (l->in_ids.n > 1) {
		for (size_t b = 0; b < batch_size; b++) {
			size_t bwh = b * wh;
			for (size_t a = 0; a < l->in_ids.n; a++) {
				layer* inl = l->in_layers[a];
				float* inl_grads = inl->grads;
				size_t inl_out_c = inl->out_c;
				size_t b_offset = bwh * inl_out_c;
				launch_backward_avgpool_kernel(&inl_grads[b_offset], grads, (int)wh, (int)inl_out_c, 1);
				grads += inl_out_c;
			}
		}
	}
	else launch_backward_avgpool_kernel(l->in_layers[0]->grads, grads, (int)wh, (int)l->c, batch_size);
}

/*** TESTING ***/

void test_forward_avgpool(void) {
	layer l = { 0 };
	network net = { 0 };
	net.batch_size = 2;
	l.w = 2;
	l.h = 2;
	
	layer inl1 = { 0 };
	layer inl2 = { 0 };
	inl1.out_w = l.w;
	inl2.out_w = l.w;
	inl1.out_h = l.h;
	inl2.out_h = l.h;
	inl1.out_c = 1;
	inl2.out_c = 2;
	inl1.out_n = inl1.out_w * inl1.out_h * inl1.out_c;
	inl2.out_n = inl2.out_w * inl2.out_h * inl2.out_c;
	inl1.output = (float*)xcalloc(inl1.out_n * net.batch_size, sizeof(float));
	//fill_array_rand_float(inl1.output, inl1.out_n * net.batch_size, 0.0, 1.0);
	fill_array_increment(inl1.output, inl1.out_n * net.batch_size, 0.0F, 2.0F);
	inl2.output = (float*)xcalloc(inl2.out_n * net.batch_size, sizeof(float));
	//fill_array_rand_float(inl2.output, inl2.out_n * net.batch_size, 0.0, 1.0);
	fill_array_increment(inl2.output, inl2.out_n * net.batch_size, 0.0F, 1.0F);

	l.c = inl1.out_c + inl2.out_c;
	l.n = l.w * l.h * l.c;
	l.out_w = 1;
	l.out_h = 1;
	l.out_c = l.c;
	l.out_n = l.out_w * l.out_h * l.out_c;
	l.output = (float*)xcalloc(l.out_n * net.batch_size, sizeof(float));
	l.Z = l.output;

	inl1.grads = (float*)xcalloc(inl1.out_n * net.batch_size, sizeof(float));
	inl2.grads = (float*)xcalloc(inl2.out_n * net.batch_size, sizeof(float));
	l.in_layers = (layer**)xcalloc(2, sizeof(layer*));
	l.in_layers[0] = &inl1;
	l.in_layers[1] = &inl2;
	l.in_ids.n = 2;
	forward_avgpool(&l, &net);

	pprint_mat(inl1.output, (int)inl1.out_w, (int)inl1.out_h, (int)inl1.out_c * (int)net.batch_size);
	pprint_mat(inl2.output, (int)inl2.out_w, (int)inl2.out_h, (int)inl2.out_c * (int)net.batch_size);
	pprint_mat(l.output, (int)l.out_w, (int)l.out_h, (int)l.out_c * (int)net.batch_size);
}

void test_backward_avgpool(void) {
	layer l = { 0 };
	network net = { 0 };
	net.batch_size = 2;
	l.w = 2;
	l.h = 2;

	layer inl1 = { 0 };
	layer inl2 = { 0 };
	inl1.out_w = l.w;
	inl2.out_w = l.w;
	inl1.out_h = l.h;
	inl2.out_h = l.h;
	inl1.out_c = 1;
	inl2.out_c = 2;
	inl1.out_n = inl1.out_w * inl1.out_h * inl1.out_c;
	inl2.out_n = inl2.out_w * inl2.out_h * inl2.out_c;
	inl1.output = (float*)xcalloc(inl1.out_n * net.batch_size, sizeof(float));
	//fill_array_rand_float(inl1.output, inl1.out_n * net.batch_size, 0.0, 1.0);
	fill_array_increment(inl1.output, inl1.out_n * net.batch_size, 0.0F, 2.0F);
	inl2.output = (float*)xcalloc(inl2.out_n * net.batch_size, sizeof(float));
	//fill_array_rand_float(inl2.output, inl2.out_n * net.batch_size, 0.0, 1.0);
	fill_array_increment(inl2.output, inl2.out_n * net.batch_size, 0.0F, 1.0F);

	l.c = inl1.out_c + inl2.out_c;
	l.n = l.w * l.h * l.c;
	l.out_w = 1;
	l.out_h = 1;
	l.out_c = l.c;
	l.out_n = l.out_w * l.out_h * l.out_c;
	l.output = (float*)xcalloc(l.out_n * net.batch_size, sizeof(float));
	l.Z = l.output;

	inl1.grads = (float*)xcalloc(inl1.out_n * net.batch_size, sizeof(float));
	inl2.grads = (float*)xcalloc(inl2.out_n * net.batch_size, sizeof(float));
	l.in_layers = (layer**)xcalloc(2, sizeof(layer*));
	l.in_layers[0] = &inl1;
	l.in_layers[1] = &inl2;
	l.in_ids.n = 2;
	forward_avgpool(&l, &net);

	l.grads = (float*)xcalloc(l.out_n * net.batch_size, sizeof(float));

	fill_array_increment(l.grads, l.out_n * net.batch_size, 0.0F, 4.0F);
	backward_avgpool(&l, &net);
	pprint_mat(l.grads, (int)l.out_w, (int)l.out_h, (int)(l.out_c * net.batch_size));
	pprint_mat(inl1.grads, (int)inl1.out_w, (int)inl1.out_h, (int)(inl1.out_c * net.batch_size));
	pprint_mat(inl2.grads, (int)inl2.out_w, (int)inl2.out_h, (int)(inl2.out_c * net.batch_size));
}