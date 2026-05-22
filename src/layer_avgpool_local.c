#include "layer_avgpool_global.h"
#include <omp.h>
#include "derivatives.h"
#include "utils.h"
#include "xallocs.h"
#include "xcuda.h"
#include "blas.h"


// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
inline static int is_a_ge_zero_and_a_lt_b(int a, int b) {
	return (unsigned)(a) < (unsigned)(b);
}

void forward_avgpool_local(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	zero_array(l->Z, l->out_n * batch_size);
	size_t w = l->w;
	size_t h = l->h;
	size_t wh = w * h;
	size_t out_w = l->out_w;
	size_t out_h = l->out_h;
	size_t out_wh = out_w * out_h;
	size_t stride = l->stride;
	size_t ksize = l->ksize;
	float divisor = (float)(ksize * ksize);
	float* l_output = l->Z;
	for (size_t b = 0; b < batch_size; b++) {
		size_t bwh = b * wh;
		for (size_t i = 0; i < l->in_ids.n; i++) {
			layer* inl = l->in_layers[i];
			size_t inl_c = inl->out_c;
			float* inl_output = &inl->output[bwh * inl_c];
			size_t ch;
#pragma omp parallel for
			for (ch = 0; ch < inl_c; ch++) {
				float* input_ch = &inl_output[ch * wh];
				float* output_ch = &l_output[ch * out_wh];
				for (int krow = 0; krow < ksize; krow++) {
					for (int kcol = 0; kcol < ksize; kcol++) {
						float* output = output_ch;
						int in_row = krow;
						for (size_t out_rows = out_h; out_rows; out_rows--) {
							int in_col = kcol;
							int r = in_row * w;
							for (size_t out_cols = out_w; out_cols; out_cols--) {
								*output += input_ch[r + in_col];
								output++;
								in_col += stride;
							}
							in_row += stride;
						}
					}
				}
				for (size_t s = 0; s < out_wh; s++) {
					output_ch[s] /= divisor;
				}
			}
			l_output += inl_c * out_wh;
		}
	}
	if (l->activation) l->activate(l->Z, l->output, l->out_n, net->batch_size);
	if (net->training) zero_array(l->grads, l->out_n * batch_size);
}

void backward_avgpool_local(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	if (l->activation) get_activation_grads(l, batch_size);
	size_t w = l->w;
	size_t h = l->h;
	size_t wh = w * h;
	size_t out_w = l->out_w;
	size_t out_h = l->out_h;
	size_t out_wh = l->out_w;
	size_t ksize = l->ksize;
	size_t stride = l->stride;
	float divisor = (float)(ksize * ksize);
	float* l_grads = l->grads;
	scale_array(l_grads, batch_size * l->out_n, 1.0F / divisor);
	for (size_t b = 0; b < batch_size; b++) {
		size_t bwh = b * wh;
		for (size_t a = 0; a < l->in_ids.n; a++) {
			layer* inl = l->in_layers[a];
			size_t inl_c = inl->out_c;
			float* inl_grads = &inl->grads[bwh * inl_c];
			size_t ch;
#pragma omp parallel for
			for (ch = 0; ch < inl_c; ch++) {
				float* inl_grads_ch = &inl_grads[ch * wh];
				float* grads_ch = &l_grads[ch * out_wh];
				for (int krow = 0; krow < ksize; krow++) {
					for (int kcol = 0; kcol < ksize; kcol++) {
						float* grads = grads_ch;
						int in_row = krow;
						for (size_t out_rows = out_h; out_rows; out_rows--) {
							int in_col = kcol;
							int r = in_row * w;
							for (size_t out_cols = out_w; out_cols; out_cols--) {
								inl_grads_ch[r + in_col] += *grads;
								grads++;
								in_col += stride;
							}
							in_row += stride;
						}
					}
				}
			}
			l_grads += inl_c * out_wh;
		}
	}
}

void forward_avgpool_local_general(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	zero_array(l->Z, l->out_n * batch_size);
	size_t w = l->w;
	size_t h = l->h;
	size_t wh = w * h;
	size_t out_w = l->out_w;
	size_t out_h = l->out_h;
	size_t out_wh = out_w * out_h;
	size_t stride = l->stride;
	size_t ksize = l->ksize;
	size_t pad = l->pad;
	float divisor = (float)(ksize * ksize);
	float* l_output = l->Z;
	for (size_t b = 0; b < batch_size; b++) {
		size_t bwh = b * wh;
		for (size_t i = 0; i < l->in_ids.n; i++) {
			layer* inl = l->in_layers[i];
			size_t inl_c = inl->out_c;
			float* inl_output = &inl->output[bwh * inl_c];
			size_t ch;
#pragma omp parallel for
			for (ch = 0; ch < inl_c; ch++) {
				float* input_ch = &inl_output[ch * wh];
				float* output_ch = &l_output[ch * out_wh];
				for (int krow = 0; krow < ksize; krow++) {
					for (int kcol = 0; kcol < ksize; kcol++) {
						float* output = output_ch;
						int in_row = krow - pad;
						for (size_t out_rows = out_h; out_rows; out_rows--) {
							if (!is_a_ge_zero_and_a_lt_b(in_row, h)) {
								output += out_w;
							}
							else {
								int in_col = kcol - pad;
								int r = in_row * w;
								for (size_t out_cols = out_w; out_cols; out_cols--) {
									if (is_a_ge_zero_and_a_lt_b(in_col, w)) {
										*output += input_ch[r + in_col];
									}
									output++;
									in_col += stride;
								}
							}
							in_row += stride;
						}
					}
				}
				for (size_t s = 0; s < out_wh; s++) {
					output_ch[s] /= divisor;
				}
			}
			l_output += inl_c * out_wh;
		}
	}
	if (l->activation) l->activate(l->Z, l->output, l->out_n, net->batch_size);
	if (net->training) zero_array(l->grads, l->out_n * batch_size);
}

void backward_avgpool_local_general(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	if (l->activation) get_activation_grads(l, batch_size);
	size_t w = l->w;
	size_t h = l->h;
	size_t wh = w * h;
	size_t out_w = l->out_w;
	size_t out_h = l->out_h;
	size_t out_wh = l->out_w;
	size_t ksize = l->ksize;
	size_t stride = l->stride;
	size_t pad = l->pad;
	float divisor = (float)(ksize * ksize);
	float* l_grads = l->grads;
	scale_array(l_grads, batch_size * l->out_n, 1.0F / divisor);
	for (size_t b = 0; b < batch_size; b++) {
		size_t bwh = b * wh;
		for (size_t a = 0; a < l->in_ids.n; a++) {
			layer* inl = l->in_layers[a];
			size_t inl_c = inl->out_c;
			float* inl_grads = &inl->grads[bwh * inl_c];
			size_t ch;
#pragma omp parallel for
			for (ch = 0; ch < inl_c; ch++) {
				float* inl_grads_ch = &inl_grads[ch * wh];
				float* grads_ch = &l_grads[ch * out_wh];
				for (int krow = 0; krow < ksize; krow++) {
					for (int kcol = 0; kcol < ksize; kcol++) {
						float* grads = grads_ch;
						int in_row = krow - pad;
						for (size_t out_rows = out_h; out_rows; out_rows--) {
							if (!is_a_ge_zero_and_a_lt_b(in_row, h)) {
								grads += out_w;
							}
							else {
								int in_col = kcol - pad;
								int r = in_row * w;
								for (size_t out_cols = out_w; out_cols; out_cols--) {
									if (is_a_ge_zero_and_a_lt_b(in_col, w)) {
										inl_grads_ch[r + in_col] += *grads;
									}
									grads++;
									in_col += stride;
								}
							}
							in_row += stride;
						}
					}
				}
			}
			l_grads += inl_c * out_wh;
		}
	}
}

void forward_avgpool_local_gpu(layer* l, network* net) {

}

void backward_avgpool_local_gpu(layer* l, network* net) {

}