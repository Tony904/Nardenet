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

#pragma warning(suppress:4100)  // unreferenced formal parameter: 'net'
void forward_maxpool(layer* l, network* net) {
	int w = (int)l->w;
	int h = (int)l->h;
	int wh = w * h;
	int stride = (int)l->stride;
	int ksize = (int)l->ksize;
	int pad = (int)l->pad;
	int out_w = (int)l->out_w;
	int out_h = (int)l->out_h;
	int out_wh = out_w * out_h;
	float* B = l->output;  // out_w * out_h * out_c (out_c = in_c)
	fill_array(B, l->out_n, -FLT_MAX);
	for (size_t i = 0; i < l->in_ids.n; i++) {
		layer* inl = l->in_layers[i];
		int channels = (int)inl->out_c;
		float* A = inl->output;  // w * h * channels
		int ch;
#pragma omp parallel for firstprivate(w, h, out_w, out_h, ksize, pad, stride)
		for (ch = 0; ch < channels; ch++) {
			float* inpool = &A[ch * wh];
			float* maxpool_channel_start = &B[ch * out_wh];
			for (int krow = 0; krow < ksize; krow++) {
				for (int kcol = 0; kcol < ksize; kcol++) {
					float* maxpool = maxpool_channel_start;
					int in_row = krow - pad;
					for (int out_rows = out_h; out_rows; out_rows--) {
						if (!is_a_ge_zero_and_a_lt_b(in_row, h)) {
							maxpool += out_w;
						}
						else {
							int in_col = kcol - pad;
							for (int out_cols = out_w; out_cols; out_cols--) {
								if (is_a_ge_zero_and_a_lt_b(in_col, w)) {
									float val = inpool[in_row * w + in_col];
									if (val > *maxpool) *maxpool = val;
								}
								maxpool++;
								in_col += stride;
							}
						}
						in_row += stride;
					}
				}
			}
		}
		B += out_w * out_h * channels;
	}
}

#pragma warning(suppress:4100)  // unreferenced formal parameter: 'net'
void backward_maxpool(layer* l, network* net) {

}

#pragma warning(suppress:4100)  // unreferenced formal parameter: 'net'
void update_maxpool(layer* l, network* net) {
	l;
}

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
}