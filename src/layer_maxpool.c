#include "layer_maxpool.h"
#include <omp.h>
#include <float.h>
#include <stdlib.h>
#include <assert.h>
#include "utils.h"
#include "xallocs.h"


// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
inline static int is_a_ge_zero_and_a_lt_b(int a, int b) {
	return (unsigned)(a) < (unsigned)(b);
}

// TODO: JUST RECORD THE ACTUAL ADDRESSES OF THE INPUT INDEXES
//		 INSTEAD OF STORING THE INDEXES.
#pragma warning(suppress:4100)  // unreferenced formal parameter: 'net'
void forward_maxpool(layer* l, network* net) {
	float** max_addresses = l->maxpool_addresses;
	int w = (int)l->w;
	int h = (int)l->h;
	size_t wh = (size_t)w * (size_t)h;
	int stride = (int)l->stride;
	int ksize = (int)l->ksize;
	int pad = (int)l->pad;
	size_t out_w = l->out_w;
	size_t out_h = l->out_h;
	size_t out_wh = out_w * out_h;
	float* B = l->output;  // out_w * out_h * out_c (out_c = in_c)
	fill_array(B, l->out_n, -FLT_MAX);
	for (size_t i = 0; i < l->in_ids.n; i++) {
		layer* inl = l->in_layers[i];
		size_t channels = (int)inl->out_c;
		float* A = inl->output;  // w * h * channels
		size_t ch;
#pragma omp parallel for firstprivate(w, h, wh, out_w, out_h, out_wh, ksize, pad, stride)
		for (ch = 0; ch < channels; ch++) {
			float* inpool = &A[ch * wh];
			size_t out_whc = ch * out_wh;
			float* maxpool_channel_start = &B[out_whc];
			float** maxes_channel_start = &max_addresses[out_whc];
			for (int krow = 0; krow < ksize; krow++) {
				for (int kcol = 0; kcol < ksize; kcol++) {
					float* maxpool = maxpool_channel_start;
					float** maxes = maxes_channel_start;
					int in_row = krow - pad;
					for (size_t out_rows = out_h; out_rows; out_rows--) {
						if (!is_a_ge_zero_and_a_lt_b(in_row, h)) {
							maxpool += out_w;
							maxes += out_w;
						}
						else {
							int in_col = kcol - pad;
							int r = in_row * w;
							for (size_t out_cols = out_w; out_cols; out_cols--) {
								if (is_a_ge_zero_and_a_lt_b(in_col, w)) {
									size_t index = (size_t)(r + in_col);
									float val = inpool[index];
									if (val > *maxpool) {
										*maxpool = val;
										*maxes = &inpool[index];
									}
								}
								maxpool++;
								maxes++;
								in_col += stride;
							}
						}
						in_row += stride;
					}
				}
			}
		}
		B += out_wh * channels;
		max_addresses += out_wh * channels;
	}
}

#pragma warning(suppress:4100)  // unreferenced formal parameter: 'net'
void backward_maxpool(layer* l, network* net) {
	for (size_t i = 0; i < l->in_ids.n; i++) {
		layer* inl = l->in_layers[i];
		zero_array(inl->output, inl->out_n);
	}
	float* grads = l->output;
	float** maxes = l->maxpool_addresses;
	size_t out_n = l->out_n;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < out_n; i++) {
		*maxes[i] = grads[i];
	}
}


/*** TESTING ***/


void test_forward_maxpool(void) {
	layer* l = (layer*)xcalloc(1, sizeof(layer));
	network* net = (network*)xcalloc(1, sizeof(network));
	l->ksize = 2;
	l->pad = 0;
	l->stride = 2;
	l->w = 4;
	l->h = 4;
	l->c = 3;
	l->out_w = (l->w + 2 * l->pad - l->ksize) / l->stride + 1;
	l->out_h = (l->h + 2 * l->pad - l->ksize) / l->stride + 1;
	l->out_c = l->c;
	l->out_n = l->out_w * l->out_h * l->out_c;
	l->output = (float*)xcalloc(l->out_n, sizeof(float));
	l->maxpool_addresses = (float**)xcalloc(l->out_n, sizeof(float*));
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
	fill_array_rand_float(inl1->output, inl1->out_n, 0.0F, 100.0F);
	inl2->output = (float*)xcalloc(inl2->out_n, sizeof(float));
	fill_array_rand_float(inl2->output, inl2->out_n, 25.0F, 75.0F);
	l->in_layers = (layer**)xcalloc(2, sizeof(layer*));
	l->in_layers[0] = inl1;
	l->in_layers[1] = inl2;
	l->in_ids.n = 2;

	forward_maxpool(l, net);

	pprint_mat(inl1->output, (int)inl1->out_w, (int)inl1->out_h, (int)inl1->out_c);
	pprint_mat(inl2->output, (int)inl2->out_w, (int)inl2->out_h, (int)inl2->out_c);
	pprint_mat(l->output, (int)l->out_w, (int)l->out_h, (int)l->out_c);
	for (int i = 0; i < l->out_n; i++) {
		printf("%f\n", *l->maxpool_addresses[i]);
	}
}

void test_backward_maxpool(void) {
	layer* l = (layer*)xcalloc(1, sizeof(layer));
	network* net = (network*)xcalloc(1, sizeof(network));
	l->ksize = 2;
	l->pad = 0;
	l->stride = 2;
	l->w = 4;
	l->h = 4;
	l->c = 3;
	l->out_w = (l->w + 2 * l->pad - l->ksize) / l->stride + 1;
	l->out_h = (l->h + 2 * l->pad - l->ksize) / l->stride + 1;
	l->out_c = l->c;
	l->out_n = l->out_w * l->out_h * l->out_c;
	l->output = (float*)xcalloc(l->out_n, sizeof(float));
	l->maxpool_addresses = (float**)xcalloc(l->out_n, sizeof(float*));
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
	fill_array_rand_float(inl1->output, inl1->out_n, 0.0F, 100.0F);
	inl2->output = (float*)xcalloc(inl2->out_n, sizeof(float));
	fill_array_rand_float(inl2->output, inl2->out_n, 25.0F, 75.0F);
	l->in_layers = (layer**)xcalloc(2, sizeof(layer*));
	l->in_layers[0] = inl1;
	l->in_layers[1] = inl2;
	l->in_ids.n = 2;

	forward_maxpool(l, net);

	pprint_mat(inl1->output, (int)inl1->out_w, (int)inl1->out_h, (int)inl1->out_c);
	pprint_mat(inl2->output, (int)inl2->out_w, (int)inl2->out_h, (int)inl2->out_c);
	pprint_mat(l->output, (int)l->out_w, (int)l->out_h, (int)l->out_c);
	for (int i = 0; i < l->out_n; i++) {
		printf("%f\n", *l->maxpool_addresses[i]);
	}
}