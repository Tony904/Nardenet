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

/* Standard maxpool operation with ksize = 2, pad = 0, stride = 2 */
void forward_maxpool_standard(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	size_t batch_size = net->batch_size;
	float** l_max_ptrs = l->maxpool_addresses;
	float* l_output = l->output;
	fill_array(l_output, out.n * batch_size, -FLT_MAX);
	for (size_t b = 0; b < batch_size; b++) {
		float* b_output = &l_output[b * out.n];
		float** b_max_ptrs = &l_max_ptrs[b * out.n];
		for (size_t i = 0; i < l->n_inputs; i++) {
			layer_input input = l->inputs[i];
			float* inl_output = &input.data[b * input.stride];
			float* inl_grads = &input.grads[b * input.stride];
			size_t ch;
#pragma omp parallel for
			for (ch = 0; ch < input.c; ch++) {
				size_t inl_ch_start = ch * in.sp;
				size_t l_ch_start = ch * out.sp;
				float* A = &inl_output[inl_ch_start];
				float* G = &inl_grads[inl_ch_start];
				float* max_vals_ch_start = &b_output[l_ch_start];
				float** max_ptrs_ch_start = &b_max_ptrs[l_ch_start];
				for (size_t krow = 0; krow < k.ksize; krow++) {
					for (size_t kcol = 0; kcol < k.ksize; kcol++) {
						float* max_vals = max_vals_ch_start;
						float** max_ptrs = max_ptrs_ch_start;
						size_t in_row = krow;
						for (size_t out_rows = out.h; out_rows; out_rows--) {
							size_t in_col = kcol;
							size_t r = in_row * in.w;
							for (size_t out_cols = out.w; out_cols; out_cols--) {
								size_t index = (size_t)(r + in_col);
								float val = A[index];
								if (val > *max_vals) {
									*max_vals = val;
									*max_ptrs = &G[index];
								}
								max_vals++;
								max_ptrs++;
								in_col += k.stride;
							}
							in_row += k.stride;
						}
					}
				}
			}
			// shift pointers by the size of the output of the input layer that was just processed
			b_output += out.sp * input.c;
			b_max_ptrs += out.sp * input.c;
		}
	}
	if (net->training) zero_array(l->grads, out.n * batch_size);
}

/* Version of maxpool that allows for any ksize, pad, and stride. */
void forward_maxpool_general(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	size_t batch_size = net->batch_size;
	int ksize = (int)k.ksize;
	int pad = (int)k.pad;
	int stride = (int)k.stride;
	int w = (int)in.w;
	int h = (int)in.h;
	float** l_max_ptrs = l->maxpool_addresses;
	float* l_output = l->output;
	fill_array(l_output, out.n * batch_size, -FLT_MAX);
	for (size_t b = 0; b < batch_size; b++) {
		float* b_output = &l_output[b * out.n];
		float** b_max_ptrs = &l_max_ptrs[b * out.n];
		for (size_t i = 0; i < l->n_inputs; i++) {
			layer_input input = l->inputs[i];
			float* inl_output = &input.data[b * input.stride];
			float* inl_grads = &input.grads[b * input.stride];
			size_t ch;
#pragma omp parallel for
			for (ch = 0; ch < input.c; ch++) {
				size_t inl_ch_start = ch * in.sp;
				size_t l_ch_start = ch * out.sp;
				float* A = &inl_output[inl_ch_start];
				float* G = &inl_grads[inl_ch_start];
				float* max_vals_ch_start = &b_output[l_ch_start];
				float** max_ptrs_ch_start = &b_max_ptrs[l_ch_start];
				for (int krow = 0; krow < ksize; krow++) {
					for (int kcol = 0; kcol < ksize; kcol++) {
						float* max_vals = max_vals_ch_start;
						float** max_ptrs = max_ptrs_ch_start;
						int in_row = krow - pad;
						for (size_t out_rows = out.h; out_rows; out_rows--) {
							if (!is_a_ge_zero_and_a_lt_b(in_row, h)) {
								max_vals += out.w;
								max_ptrs += out.w;
							}
							else {
								int in_col = kcol - pad;
								int r = in_row * w;
								for (size_t out_cols = out.w; out_cols; out_cols--) {
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
			b_output += out.sp * input.c;
			b_max_ptrs += out.sp * input.c;
		}
	}
	if (net->training) zero_array(l->grads, out.n * batch_size);
}

#pragma warning(suppress:4100)  // unreferenced formal parameter: 'net'
void backward_maxpool(layer* l, network* net) {
	float* grads = l->grads;
	float** max_ptrs = l->maxpool_addresses;
	size_t n = l->out.n * net->batch_size;
	size_t i;
#pragma omp parallel for
	for (i = 0; i < n; i++) {
		*max_ptrs[i] += grads[i];
	}
}

#ifdef GPU
void forward_maxpool_gpu(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	int batch_size = (int)net->batch_size;
	int ksize = (int)k.ksize;
	int stride = (int)k.stride;
	int w = (int)in.w;
	int h = (int)in.h;
	int out_n = (int)out.n;
	int out_w = (int)out.w;
	int out_h = (int)out.h;
	int out_wh = out_w * out_h;
	float* l_output = l->gpu.output;
	float** l_max_ptrs = l->gpu.maxpool_addresses;
	for (int b = 0; b < batch_size; b++) {
		float* b_output = &l_output[b * out_n];
		float** b_max_ptrs = &l_max_ptrs[b * out_n];
		for (int i = 0; i < l->n_inputs; i++) {
			layer_input input = l->inputs[i];
			float* inl_output = &input.gpu.data[b * input.stride];
			float* inl_grads = &input.gpu.grads[b * input.stride];
			launch_forward_maxpool_general_kernel(inl_output, b_output, inl_grads, b_max_ptrs, w, h, out_w, out_h, out_n, ksize, stride);
			b_output += out_wh * input.c;
			b_max_ptrs += out_wh * input.c;
		}
	}
	if (net->training) zero_array_gpu(l->gpu.grads, out_n * batch_size);
}

void backward_maxpool_gpu(layer* l, network* net) {
	launch_backward_maxpool_kernel(l->gpu.grads, l->gpu.maxpool_addresses, (int)(l->out.n * net->batch_size));
}
#else
#pragma warning (suppress:4100)
void forward_maxpool_gpu(layer* l, network* net) {
	gpu_not_defined();
}
#pragma warning (suppress:4100)
void backward_maxpool_gpu(layer* l, network* net) {
	gpu_not_defined();
}
#endif


/*** TESTING ***/
//void test_forward_maxpool(void) {
//	layer* l = (layer*)xcalloc(1, sizeof(layer));
//	network* net = (network*)xcalloc(1, sizeof(network));
//	net->batch_size = 2;
//	l->ksize = 2;
//	l->pad = 0;
//	l->stride = 2;
//	l->w = 4;
//	l->h = 4;
//	l->c = 3;
//	l->out_w = (l->w + 2 * l->pad - l->ksize) / l->stride + 1;
//	l->out_h = (l->h + 2 * l->pad - l->ksize) / l->stride + 1;
//	l->out_c = l->c;
//	l->out_n = l->out_w * l->out_h * l->out_c;
//	l->output = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
//	l->maxpool_addresses = (float**)xcalloc(l->out_n * net->batch_size, sizeof(float*));
//	layer* inl1 = (layer*)xcalloc(1, sizeof(layer));
//	layer* inl2 = (layer*)xcalloc(1, sizeof(layer));
//	inl1->out_w = l->w;
//	inl2->out_w = l->w;
//	inl1->out_h = l->h;
//	inl2->out_h = l->h;
//	inl1->out_c = 1;
//	inl2->out_c = 2;
//	assert(l->c == inl1->out_c + inl2->out_c);
//	inl1->out_n = inl1->out_w * inl1->out_h * inl1->out_c;
//	inl2->out_n = inl2->out_w * inl2->out_h * inl2->out_c;
//	inl1->output = (float*)xcalloc(inl1->out_n * net->batch_size, sizeof(float));
//	fill_array_rand_float(inl1->output, inl1->out_n * net->batch_size, 0.0, (double)(inl1->out_n * net->batch_size));
//	inl2->output = (float*)xcalloc(inl2->out_n * net->batch_size, sizeof(float));
//	fill_array_rand_float(inl2->output, inl2->out_n * net->batch_size, 0.0, (double)(inl2->out_n * net->batch_size));
//	inl1->grads = (float*)xcalloc(inl1->out_n * net->batch_size, sizeof(float));
//	inl2->grads = (float*)xcalloc(inl2->out_n * net->batch_size, sizeof(float));
//	l->in_layers = (layer**)xcalloc(2, sizeof(layer*));
//	l->in_layers[0] = inl1;
//	l->in_layers[1] = inl2;
//	l->n_inputs = 2;
//	forward_maxpool(l, net);
//
//	pprint_mat(inl1->output, (int)inl1->out_w, (int)inl1->out_h, (int)inl1->out_c * (int)net->batch_size);
//	pprint_mat(inl2->output, (int)inl2->out_w, (int)inl2->out_h, (int)inl2->out_c * (int)net->batch_size);
//	pprint_mat(l->output, (int)l->out_w, (int)l->out_h, (int)l->out_c * (int)net->batch_size);
//}
//
//void test_backward_maxpool(void) {
//	layer* l = (layer*)xcalloc(1, sizeof(layer));
//	network* net = (network*)xcalloc(1, sizeof(network));
//	net->training = 1;
//	net->batch_size = 1;
//	l->ksize = 2;
//	l->pad = 0;
//	l->stride = 2;
//	l->w = 4;
//	l->h = 4;
//	l->c = 3;
//	l->out_w = (l->w + 2 * l->pad - l->ksize) / l->stride + 1;
//	l->out_h = (l->h + 2 * l->pad - l->ksize) / l->stride + 1;
//	l->out_c = l->c;
//	l->out_n = l->out_w * l->out_h * l->out_c;
//	l->grads = (float*)xcalloc(l->out_n, sizeof(float));
//	l->output = (float*)xcalloc(l->out_n, sizeof(float));
//	l->maxpool_addresses = (float**)xcalloc(l->out_n, sizeof(float*));
//	layer* inl1 = (layer*)xcalloc(1, sizeof(layer));
//	layer* inl2 = (layer*)xcalloc(1, sizeof(layer));
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
//	inl1->grads = (float*)xcalloc(inl1->out_n, sizeof(float));
//	fill_array_rand_float(inl1->output, inl1->out_n, 0.0, 100.0);
//	inl2->output = (float*)xcalloc(inl2->out_n, sizeof(float));
//	inl2->grads = (float*)xcalloc(inl2->out_n, sizeof(float));
//	fill_array_rand_float(inl2->output, inl2->out_n, 25.0, 75.0);
//	l->in_layers = (layer**)xcalloc(2, sizeof(layer*));
//	l->in_layers[0] = inl1;
//	l->in_layers[1] = inl2;
//	l->n_inputs = 2;
//
//	forward_maxpool(l, net);
//	pprint_mat(inl1->output, (int)inl1->out_w, (int)inl1->out_h, (int)inl1->out_c);
//	pprint_mat(inl2->output, (int)inl2->out_w, (int)inl2->out_h, (int)inl2->out_c);
//
//	fill_array(l->grads, l->out_n, 1.0F);
//	backward_maxpool(l, net);
//	pprint_mat(inl1->grads, (int)inl1->out_w, (int)inl1->out_h, (int)inl1->out_c);
//	pprint_mat(inl2->grads, (int)inl2->out_w, (int)inl2->out_h, (int)inl2->out_c);
//}