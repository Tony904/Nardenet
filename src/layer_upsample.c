#include "layer_upsample.h"
#include "utils.h"
#include "derivatives.h"
#include "xallocs.h"
#include "blas.h"
#include "xcuda.h"



void forward_upsample_gpu(layer* l, network* net) {
	float* Z = l->Z;
	int ksize = (int)l->ksize;
	int w = (int)l->w;
	int h = (int)l->h;
	int out_w = (int)l->out_w;
	int out_wh = out_w * (int)l->out_h;
	size_t batch_size = net->batch_size;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t a = 0; a < l->in_ids.n; a++) {
			layer* inl = l->in_layers[a];
			float* inl_output = inl->output;
			int inl_out_c = (int)inl->out_c;
			launch_forward_upsample_kernel(inl_output, Z, w, h, inl_out_c, ksize, 1);
			Z += out_wh * inl_out_c;
		}
	}
	if (l->activation) l->activate(l->Z, l->output, l->out_n, batch_size);
	if (net->training) zero_array_gpu(l->grads, (int)(l->out_n * batch_size));
}

void forward_upsample(layer* l, network* net) {
	float* Z = l->Z;
	size_t ksize = l->ksize;
	size_t w = l->w;
	size_t h = l->h;
	size_t wh = w * h;
	size_t out_w = l->out_w;
	size_t out_wh = out_w * l->out_h;
	size_t batch_size = net->batch_size;
	for (size_t b = 0; b < batch_size; b++) {
		size_t bwh = b * wh;
		for (size_t a = 0; a < l->in_ids.n; a++) {
			layer* inl = l->in_layers[a];
			float* inl_output = inl->output;
			size_t inl_out_c = inl->out_c;
			size_t b_offset = bwh * inl_out_c;
			size_t ch;
#pragma omp parallel for firstprivate(w, h, ksize)
			for (ch = 0; ch < inl_out_c; ch++) {
				size_t ch_offset = ch * wh + b_offset;
				size_t z_offset = ch * out_wh;
				for (size_t row = 0; row < h; row++) {
					size_t row_offset = ch_offset + row * w;
					size_t out_row0 = row * ksize * out_w;
					for (size_t col = 0; col < w; col++) {
						size_t out_col0 = col * ksize;
						float val = inl_output[row_offset + col];
						for (size_t krow = 0; krow < ksize; krow++) {
							size_t out_row = out_row0 + krow * out_w;
							for (size_t kcol = 0; kcol < ksize; kcol++) {
								Z[z_offset + out_row + out_col0 + kcol] = val;
							}
						}
					}
				}
			}
			Z += out_wh * inl_out_c;
		}
	}
	if (l->activation) l->activate(l->Z, l->output, l->out_n, net->batch_size);
	if (net->training) zero_array(l->grads, l->out_n * batch_size);
}

void backward_upsample_gpu(layer* l, network* net) {
	int batch_size = (int)net->batch_size;
	float* grads = l->grads;
	if (l->activation) get_activation_grads_gpu(l, batch_size);
	int ksize = (int)l->ksize;
	int w = (int)l->w;
	int h = (int)l->h;
	int out_w = (int)l->out_w;
	int out_wh = out_w * (int)l->out_h;
	for (int b = 0; b < batch_size; b++) {
		for (int a = 0; a < l->in_ids.n; a++) {
			layer* inl = l->in_layers[a];
			float* inl_grads = inl->grads;
			int inl_out_c = (int)inl->out_c;
			launch_backward_upsample_kernel(inl_grads, grads, w, h, inl_out_c, ksize, 1);
			grads += out_wh * inl_out_c;
		}
	}
}

void backward_upsample(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->grads;
	if (l->activation) get_activation_grads(l, batch_size);
	size_t ksize = l->ksize;
	size_t w = l->w;
	size_t h = l->h;
	size_t wh = w * h;
	size_t out_w = l->out_w;
	size_t out_wh = out_w * l->out_h;
	for (size_t b = 0; b < batch_size; b++) {
		size_t bwh = b * wh;
		for (size_t a = 0; a < l->in_ids.n; a++) {
			layer* inl = l->in_layers[a];
			float* inl_grads = inl->grads;
			size_t inl_out_c = inl->out_c;
			size_t b_offset = bwh * inl_out_c;
			size_t ch;
#pragma omp parallel for firstprivate(w, h, ksize)
			for (ch = 0; ch < inl_out_c; ch++) {
				size_t ch_offset = ch * wh + b_offset;
				size_t z_offset = ch * out_wh;
				for (size_t row = 0; row < h; row++) {
					size_t row_offset = ch_offset + row * w;
					size_t out_row0 = row * ksize * out_w;
					for (size_t col = 0; col < w; col++) {
						size_t out_col0 = col * ksize;
						size_t index = row_offset + col;
						for (size_t krow = 0; krow < ksize; krow++) {
							size_t out_row = out_row0 + krow * out_w;
							for (size_t kcol = 0; kcol < ksize; kcol++) {
								inl_grads[index] += grads[z_offset + out_row + out_col0 + kcol];
							}
						}
					}
				}
			}
			grads += out_wh * inl_out_c;
		}
	}
}

/*** TESTING ***/

void test_forward_upsample(void) {
	layer l = { 0 };
	network net = { 0 };
	net.batch_size = 2;
	l.w = 2;
	l.h = 2;
	l.ksize = 2;

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
	fill_array_increment(inl1.output, inl1.out_n * net.batch_size, 1.0F, 2.0F);
	inl2.output = (float*)xcalloc(inl2.out_n * net.batch_size, sizeof(float));
	fill_array_increment(inl2.output, inl2.out_n * net.batch_size, 0.5F, 1.0F);

	l.c = inl1.out_c + inl2.out_c;
	l.n = l.w * l.h * l.c;
	l.out_w = l.w * l.ksize;
	l.out_h = l.h * l.ksize;
	l.out_c = l.c;
	l.out_n = l.out_w * l.out_h * l.out_c;
	l.output = (float*)xcalloc(l.out_n * net.batch_size, sizeof(float));
	l.Z = l.output;

	l.in_layers = (layer**)xcalloc(2, sizeof(layer*));
	l.in_layers[0] = &inl1;
	l.in_layers[1] = &inl2;
	l.in_ids.n = 2;
	forward_upsample(&l, &net);

	pprint_mat_batch(inl1.output, inl1.out_w, inl1.out_h, inl1.out_c, net.batch_size);
	pprint_mat_batch(inl2.output, inl2.out_w, inl2.out_h, inl2.out_c, net.batch_size);
	pprint_mat_batch(l.output, l.out_w, l.out_h, l.out_c, net.batch_size);
}

void test_backward_upsample(void) {
	layer l = { 0 };
	network net = { 0 };
	net.batch_size = 2;
	l.w = 4;
	l.h = 4;
	l.ksize = 2;

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
	inl1.grads = (float*)xcalloc(inl1.out_n * net.batch_size, sizeof(float));
	inl2.grads = (float*)xcalloc(inl2.out_n * net.batch_size, sizeof(float));

	l.c = inl1.out_c + inl2.out_c;
	l.n = l.w * l.h * l.c;
	l.out_w = l.w * l.ksize;
	l.out_h = l.h * l.ksize;
	l.out_c = l.c;
	l.out_n = l.out_w * l.out_h * l.out_c;
	l.grads = (float*)xcalloc(l.out_n * net.batch_size, sizeof(float));

	fill_array_increment(l.grads, l.out_n * net.batch_size, 1.0F, 1.0F);
	l.in_layers = (layer**)xcalloc(2, sizeof(layer*));
	l.in_layers[0] = &inl1;
	l.in_layers[1] = &inl2;
	l.in_ids.n = 2;
	backward_upsample(&l, &net);


	pprint_mat_batch(l.grads, l.out_w, l.out_h, l.out_c, net.batch_size);
	pprint_mat_batch(inl1.grads, inl1.out_w, inl1.out_h, inl1.out_c, net.batch_size);
	pprint_mat_batch(inl2.grads, inl2.out_w, inl2.out_h, inl2.out_c, net.batch_size);
}