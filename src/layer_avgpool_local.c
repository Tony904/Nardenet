#include "layer_avgpool_global.h"
#include <omp.h>
#include "derivatives.h"
#include "utils.h"
#include "xallocs.h"
#include "xcuda.h"
#include "blas.h"
#include "layer_testing.h"


// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
inline static int is_a_ge_zero_and_a_lt_b(int a, int b) {
	return (unsigned)(a) < (unsigned)(b);
}


void forward_avgpool_local(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	size_t batch_size = net->batch_size;
	zero_array(l->Z, out.n * batch_size);
	int w = (int)in.w;
	int stride = (int)k.stride;
	float divisor = (float)(k.ksize * k.ksize);
	float* l_output = l->Z;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t i = 0; i < l->n_inputs; i++) {
			layer_input input = l->inputs[i];
			float* inl_output = &input.data[b * input.stride];
			size_t ch;
#pragma omp parallel for
			for (ch = 0; ch < input.c; ch++) {
				float* input_ch = &inl_output[ch * in.sp];
				float* output_ch = &l_output[ch * out.sp];
				for (int krow = 0; krow < k.ksize; krow++) {
					for (int kcol = 0; kcol < k.ksize; kcol++) {
						float* output = output_ch;
						int in_row = krow;
						for (size_t out_rows = out.h; out_rows; out_rows--) {
							int in_col = kcol;
							int r = in_row * w;
							for (size_t out_cols = out.w; out_cols; out_cols--) {
								*output += input_ch[r + in_col];
								output++;
								in_col += stride;
							}
							in_row += stride;
						}
					}
				}
				for (size_t s = 0; s < out.sp; s++) {
					output_ch[s] /= divisor;
				}
			}
			l_output += input.c * out.sp;
		}
	}
	if (l->activation) l->activate(l->Z, l->output, out.n, net->batch_size);
	if (net->training) zero_array(l->grads, out.n * batch_size);
}


void backward_avgpool_local(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	size_t batch_size = net->batch_size;
	if (l->activation) get_activation_grads(l, batch_size);
	int w = (int)in.w;
	int stride = (int)k.stride;
	float divisor = (float)(k.ksize * k.ksize);
	float* l_grads = l->grads;
	scale_array(l_grads, batch_size * out.n, 1.0F / divisor);
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t a = 0; a < l->n_inputs; a++) {
			layer_input input = l->inputs[a];
			float* inl_grads = &input.grads[b * input.stride];
			size_t ch;
#pragma omp parallel for
			for (ch = 0; ch < input.c; ch++) {
				float* inl_grads_ch = &inl_grads[ch * in.sp];
				float* grads_ch = &l_grads[ch * out.sp];
				for (int krow = 0; krow < k.ksize; krow++) {
					for (int kcol = 0; kcol < k.ksize; kcol++) {
						float* grads = grads_ch;
						int in_row = krow;
						for (size_t out_rows = out.h; out_rows; out_rows--) {
							int in_col = kcol;
							int r = in_row * w;
							for (size_t out_cols = out.w; out_cols; out_cols--) {
								inl_grads_ch[r + in_col] += *grads;
								grads++;
								in_col += stride;
							}
							in_row += stride;
						}
					}
				}
			}
			l_grads += input.c * out.sp;
		}
	}
}


void forward_avgpool_local_general(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	size_t batch_size = net->batch_size;
	zero_array(l->Z, out.n * batch_size);
	int w = (int)in.w;
	int h = (int)in.h;
	int pad = (int)k.pad;
	int stride = (int)k.stride;
	float divisor = (float)(k.ksize * k.ksize);
	float* l_output = l->Z;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t i = 0; i < l->n_inputs; i++) {
			layer_input input = l->inputs[i];
			float* inl_output = &input.data[b * input.stride];
			size_t ch;
#pragma omp parallel for
			for (ch = 0; ch < input.c; ch++) {
				float* input_ch = &inl_output[ch * in.sp];
				float* output_ch = &l_output[ch * out.sp];
				for (int krow = 0; krow < k.ksize; krow++) {
					for (int kcol = 0; kcol < k.ksize; kcol++) {
						float* output = output_ch;
						int in_row = krow - pad;
						for (size_t out_rows = out.h; out_rows; out_rows--) {
							if (!is_a_ge_zero_and_a_lt_b(in_row, h)) {
								output += out.w;
							}
							else {
								int in_col = kcol - pad;
								int r = in_row * w;
								for (size_t out_cols = out.w; out_cols; out_cols--) {
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
				for (size_t s = 0; s < out.sp; s++) {
					output_ch[s] /= divisor;
				}
			}
			l_output += input.c * out.sp;
		}
	}
	if (l->activation) l->activate(l->Z, l->output, out.n, net->batch_size);
	if (net->training) zero_array(l->grads, out.n * batch_size);
}


void backward_avgpool_local_general(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	size_t batch_size = net->batch_size;
	if (l->activation) get_activation_grads(l, batch_size);
	int w = (int)in.w;
	int h = (int)in.h;
	int pad = (int)k.pad;
	int stride = (int)k.stride;
	float divisor = (float)(k.ksize * k.ksize);
	float* l_grads = l->grads;
	scale_array(l_grads, batch_size * out.n, 1.0F / divisor);
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t a = 0; a < l->n_inputs; a++) {
			layer_input input = l->inputs[a];
			float* inl_grads = &input.grads[b * input.stride];
			size_t ch;
#pragma omp parallel for
			for (ch = 0; ch < input.c; ch++) {
				float* inl_grads_ch = &inl_grads[ch * in.sp];
				float* grads_ch = &l_grads[ch * out.sp];
				for (int krow = 0; krow < k.ksize; krow++) {
					for (int kcol = 0; kcol < k.ksize; kcol++) {
						float* grads = grads_ch;
						int in_row = krow - pad;
						for (size_t out_rows = out.h; out_rows; out_rows--) {
							if (!is_a_ge_zero_and_a_lt_b(in_row, h)) {
								grads += out.w;
							}
							else {
								int in_col = kcol - pad;
								int r = in_row * w;
								for (size_t out_cols = out.w; out_cols; out_cols--) {
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
			l_grads += input.c * out.sp;
		}
	}
}


void forward_avgpool_local_gpu(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	size_t batch_size = net->batch_size;
	zero_array_gpu(l->gpu.Z, (int)(out.n * batch_size));
	int standard = ((int)k.ksize == 2 && (int)k.stride == 2) ? 1 : 0;
	float* l_output = l->gpu.Z;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t i = 0; i < l->n_inputs; i++) {
			layer_input input = l->inputs[i];
			float* inl_output = &input.gpu.data[b * input.stride];
			if (standard) launch_forward_avgpool_local_standard_kernel(inl_output, l_output, (int)in.w, (int)in.h, (int)out.w, (int)out.h, (int)out.n);
			else launch_forward_avgpool_local_general_kernel(inl_output, l_output, (int)in.w, (int)in.h, (int)out.w, (int)out.h, (int)out.n, (int)k.ksize, (int)k.stride);
			l_output += input.c * out.sp;
		}
	}
	if (l->activation) l->activate(l->gpu.Z, l->gpu.output, out.n, net->batch_size);
	if (net->training) zero_array_gpu(l->gpu.grads, (int)(out.n * batch_size));
}


void backward_avgpool_local_gpu(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	size_t batch_size = net->batch_size;
	float* l_grads = l->gpu.grads;
	if (l->activation) get_activation_grads_gpu(l, batch_size);
	int standard = (k.ksize == 2 && k.stride == 2) ? 1 : 0;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t i = 0; i < l->n_inputs; i++) {
			layer_input input = l->inputs[i];
			float* inl_grads = &input.gpu.grads[b * input.stride];
			if (standard) launch_backward_avgpool_local_standard_kernel(inl_grads, l_grads, (int)in.w, (int)in.h, (int)out.w, (int)out.h, (int)out.n);
			else launch_backward_avgpool_local_general_kernel(inl_grads, l_grads, (int)in.w, (int)in.h, (int)out.w, (int)out.h, (int)out.n, (int)k.ksize, (int)k.stride);
			l_grads += input.c * out.sp;
		}
	}
}