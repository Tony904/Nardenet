#include "layer_maxpool.h"
#include <omp.h>
#include <float.h>
#include <stdlib.h>
#include <assert.h>
#include "utils.h"
#include "xallocs.h"
#include "blas.h"
#include "xcuda.h"



// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
inline static int is_a_ge_zero_and_a_lt_b(int a, int b) {
	return (unsigned)(a) < (unsigned)(b);
}

void forward_maxpool_gpu(layer* l, network* net) {

	int batch_size = (int)net->batch_size;
	int w = (int)l->w;
	int h = (int)l->h;
	int wh = w * h;
	int out_n = (int)l->out_n;
	int out_w = (int)l->out_w;
	int out_h = (int)l->out_h;
	int out_wh = out_w * out_h;
	float* l_output = l->output;
	float** l_max_ptrs = l->maxpool_addresses;
	for (int b = 0; b < batch_size; b++) {
		float* b_output = &l_output[b * out_n];
		float** b_max_ptrs = &l_max_ptrs[b * out_n];
		int bwh = b * wh;
		for (int i = 0; i < l->in_ids.n; i++) {
			layer* inl = l->in_layers[i];
			int inl_out_c = (int)inl->out_c;
			float* inl_output = &inl->output[bwh * inl_out_c];
			//float* src, float* dst, float** max_ptrs, int src_w, int src_h, int dst_w, int dst_h, int dst_n, int batch_size
			launch_forward_maxpool_kernel(inl_output, b_output, b_max_ptrs, w, h, out_w, out_h, out_n, batch_size);
			// shift pointers by the size of the output of the input layer that was just processed
			b_output += out_wh * inl_out_c;
			b_max_ptrs += out_wh * inl_out_c;
		}
	}
	if (net->training) zero_array_gpu(l->grads, (int)(l->out_n * batch_size));
}

void backward_maxpool_gpu(layer* l, network* net) {
	launch_backward_maxpool_kernel(l->grads, l->maxpool_addresses, (int)(l->out_n * net->batch_size));
}

/* Standard maxpool operation with ksize = 2, pad = 0, stride = 2 */
void forward_maxpool(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	size_t ksize = 2;
	size_t stride = 2;
	size_t w = l->w;
	size_t h = l->h;
	size_t wh = w * h;
	size_t out_n = l->out_n;
	size_t out_w = l->out_w;
	size_t out_h = l->out_h;
	size_t out_wh = out_w * out_h;
	float** l_max_ptrs = l->maxpool_addresses;
	float* l_output = l->output;
	fill_array(l_output, out_n * batch_size, -FLT_MAX);
	for (size_t b = 0; b < batch_size; b++) {
		float* b_output = &l_output[b * out_n];
		float** b_max_ptrs = &l_max_ptrs[b * out_n];
		size_t bwh = b * wh;
		for (size_t i = 0; i < l->in_ids.n; i++) {
			layer* inl = l->in_layers[i];
			size_t inl_out_c = inl->out_c;
			float* inl_output = &inl->output[bwh * inl_out_c];
			float* inl_grads = &inl->grads[bwh * inl_out_c];
			size_t ch;
#pragma omp parallel for firstprivate(w, h, wh, out_w, out_h, out_wh, ksize, stride)
			for (ch = 0; ch < inl_out_c; ch++) {
				size_t inl_ch_start = ch * wh;
				size_t l_ch_start = ch * out_wh;
				float* A = &inl_output[inl_ch_start];
				float* G = &inl_grads[inl_ch_start];
				float* max_vals_ch_start = &b_output[l_ch_start];
				float** max_ptrs_ch_start = &b_max_ptrs[l_ch_start];
				for (size_t krow = 0; krow < ksize; krow++) {
					for (size_t kcol = 0; kcol < ksize; kcol++) {
						float* max_vals = max_vals_ch_start;
						float** max_ptrs = max_ptrs_ch_start;
						size_t in_row = krow;
						for (size_t out_rows = out_h; out_rows; out_rows--) {
							size_t in_col = kcol;
							size_t r = in_row * w;
							for (size_t out_cols = out_w; out_cols; out_cols--) {
								size_t index = (size_t)(r + in_col);
								float val = A[index];
								if (val > *max_vals) {
									*max_vals = val;
									*max_ptrs = &G[index];
								}
								max_vals++;
								max_ptrs++;
								in_col += stride;
							}
							in_row += stride;
						}
					}
				}
			}
			// shift pointers by the size of the output of the input layer that was just processed
			b_output += out_wh * inl_out_c;
			b_max_ptrs += out_wh * inl_out_c;
		}
	}
	if (net->training) zero_array(l->grads, l->out_n * batch_size);
}

/* Version of maxpool that allows for any ksize, pad, and stride. */
void forward_maxpool_general(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	int ksize = (int)l->ksize;
	int pad = (int)l->pad;
	int stride = (int)l->stride;
	int w = (int)l->w;
	int h = (int)l->h;
	size_t wh = (size_t)(w * h);
	size_t out_n = l->out_n;
	size_t out_w = l->out_w;
	size_t out_h = l->out_h;
	size_t out_wh = out_w * out_h;
	float** l_max_ptrs = l->maxpool_addresses;
	for (size_t b = 0; b < batch_size; b++) {
		float* b_output = &l->output[b * out_n];
		fill_array(b_output, out_n, -FLT_MAX);
		float** b_max_ptrs = &l_max_ptrs[b * out_n];
		size_t bwh = b * wh;
		for (size_t i = 0; i < l->in_ids.n; i++) {
			layer* inl = l->in_layers[i];
			size_t inl_out_c = inl->out_c;
			float* inl_output = &inl->output[bwh * inl_out_c];
			float* inl_grads = &inl->grads[bwh * inl_out_c];
			size_t ch;
#pragma omp parallel for firstprivate(w, h, wh, out_w, out_h, out_wh, ksize, pad, stride)
			for (ch = 0; ch < inl_out_c; ch++) {
				size_t inl_ch_start = ch * wh;
				size_t l_ch_start = ch * out_wh;
				float* A = &inl_output[inl_ch_start];
				float* G = &inl_grads[inl_ch_start];
				float* max_vals_ch_start = &b_output[l_ch_start];
				float** max_ptrs_ch_start = &b_max_ptrs[l_ch_start];
				for (int krow = 0; krow < ksize; krow++) {
					for (int kcol = 0; kcol < ksize; kcol++) {
						float* max_vals = max_vals_ch_start;
						float** max_ptrs = max_ptrs_ch_start;
						int in_row = krow - pad;
						for (size_t out_rows = out_h; out_rows; out_rows--) {
							if (!is_a_ge_zero_and_a_lt_b(in_row, h)) {
								max_vals += out_w;
								max_ptrs += out_w;
							}
							else {
								int in_col = kcol - pad;
								int r = in_row * w;
								for (size_t out_cols = out_w; out_cols; out_cols--) {
									if (is_a_ge_zero_and_a_lt_b(in_col, w)) {
										size_t index = (size_t)(r + in_col);
										float val = A[index];
										if (val > *max_vals) {
											*max_vals = val;
											*max_ptrs = &G[index];
										}
									}
									max_vals++;
									max_ptrs++;
									in_col += stride;
								}
							}
							in_row += stride;
						}
					}
				}
			}
			b_output += out_wh * inl_out_c;
			b_max_ptrs += out_wh * inl_out_c;
		}
	}
	if (net->training) zero_array(l->grads, l->out_n * batch_size);
}

#pragma warning(suppress:4100)  // unreferenced formal parameter: 'net'
void backward_maxpool(layer* l, network* net) {
	float* grads = l->grads;
	float** max_ptrs = l->maxpool_addresses;
	size_t n = l->out_n * net->batch_size;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		*max_ptrs[i] += grads[i];
	}
}


/*** TESTING ***/


void test_forward_maxpool(void) {
	layer* l = (layer*)xcalloc(1, sizeof(layer));
	network* net = (network*)xcalloc(1, sizeof(network));
	net->batch_size = 2;
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
	l->output = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	l->maxpool_addresses = (float**)xcalloc(l->out_n * net->batch_size, sizeof(float*));
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
	inl1->output = (float*)xcalloc(inl1->out_n * net->batch_size, sizeof(float));
	fill_array_rand_float(inl1->output, inl1->out_n * net->batch_size, 0.0, (double)(inl1->out_n * net->batch_size));
	inl2->output = (float*)xcalloc(inl2->out_n * net->batch_size, sizeof(float));
	fill_array_rand_float(inl2->output, inl2->out_n * net->batch_size, 0.0, (double)(inl2->out_n * net->batch_size));
	inl1->grads = (float*)xcalloc(inl1->out_n * net->batch_size, sizeof(float));
	inl2->grads = (float*)xcalloc(inl2->out_n * net->batch_size, sizeof(float));
	l->in_layers = (layer**)xcalloc(2, sizeof(layer*));
	l->in_layers[0] = inl1;
	l->in_layers[1] = inl2;
	l->in_ids.n = 2;
	forward_maxpool(l, net);

	pprint_mat(inl1->output, (int)inl1->out_w, (int)inl1->out_h, (int)inl1->out_c * (int)net->batch_size);
	pprint_mat(inl2->output, (int)inl2->out_w, (int)inl2->out_h, (int)inl2->out_c * (int)net->batch_size);
	pprint_mat(l->output, (int)l->out_w, (int)l->out_h, (int)l->out_c * (int)net->batch_size);
}

void test_backward_maxpool(void) {
	layer* l = (layer*)xcalloc(1, sizeof(layer));
	network* net = (network*)xcalloc(1, sizeof(network));
	net->training = 1;
	net->batch_size = 1;
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
	l->grads = (float*)xcalloc(l->out_n, sizeof(float));
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
	inl1->grads = (float*)xcalloc(inl1->out_n, sizeof(float));
	fill_array_rand_float(inl1->output, inl1->out_n, 0.0, 100.0);
	inl2->output = (float*)xcalloc(inl2->out_n, sizeof(float));
	inl2->grads = (float*)xcalloc(inl2->out_n, sizeof(float));
	fill_array_rand_float(inl2->output, inl2->out_n, 25.0, 75.0);
	l->in_layers = (layer**)xcalloc(2, sizeof(layer*));
	l->in_layers[0] = inl1;
	l->in_layers[1] = inl2;
	l->in_ids.n = 2;

	forward_maxpool(l, net);
	pprint_mat(inl1->output, (int)inl1->out_w, (int)inl1->out_h, (int)inl1->out_c);
	pprint_mat(inl2->output, (int)inl2->out_w, (int)inl2->out_h, (int)inl2->out_c);

	fill_array(l->grads, l->out_n, 1.0F);
	backward_maxpool(l, net);
	pprint_mat(inl1->grads, (int)inl1->out_w, (int)inl1->out_h, (int)inl1->out_c);
	pprint_mat(inl2->grads, (int)inl2->out_w, (int)inl2->out_h, (int)inl2->out_c);
}