#include "layer_upsample.h"
#include "utils.h"
#include "derivatives.h"
#include "xallocs.h"
#include "blas.h"
#include "xcuda.h"



void forward_upsample(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	float* Z = l->Z;
	size_t batch_size = net->batch_size;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t a = 0; a < l->n_inputs; a++) {
			layer_input input = l->inputs[a];
			size_t b_offset = b * input.stride;
			size_t ch;
#pragma omp parallel for
			for (ch = 0; ch < input.c; ch++) {
				size_t ch_offset = ch * in.sp + b_offset;
				size_t z_offset = ch * out.sp;
				for (size_t row = 0; row < in.h; row++) {
					size_t row_offset = ch_offset + row * in.w;
					size_t out_row0 = row * k.ksize * out.w;
					for (size_t col = 0; col < in.w; col++) {
						size_t out_col0 = col * k.ksize;
						float val = input.data[row_offset + col];
						for (size_t krow = 0; krow < k.ksize; krow++) {
							size_t out_row = out_row0 + krow * out.w;
							for (size_t kcol = 0; kcol < k.ksize; kcol++) {
								Z[z_offset + out_row + out_col0 + kcol] = val;
							}
						}
					}
				}
			}
			Z += out.sp * input.c;
		}
	}
	if (l->activation) l->activate(l->Z, l->output, out.n, net->batch_size);
	if (net->training) zero_array(l->grads, out.n * batch_size);
}

void backward_upsample(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	size_t batch_size = net->batch_size;
	float* grads = l->grads;
	if (l->activation) get_activation_grads(l, batch_size);
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t a = 0; a < l->n_inputs; a++) {
			layer_input input = l->inputs[a];
			size_t b_offset = b * input.stride;
			size_t ch;
#pragma omp parallel for
			for (ch = 0; ch < input.c; ch++) {
				size_t ch_offset = ch * in.sp + b_offset;
				size_t z_offset = ch * out.sp;
				for (size_t row = 0; row < in.h; row++) {
					size_t row_offset = ch_offset + row * in.w;
					size_t out_row0 = row * k.ksize * out.w;
					for (size_t col = 0; col < in.w; col++) {
						size_t out_col0 = col * k.ksize;
						size_t index = row_offset + col;
						for (size_t krow = 0; krow < k.ksize; krow++) {
							size_t out_row = out_row0 + krow * out.w;
							for (size_t kcol = 0; kcol < k.ksize; kcol++) {
								input.grads[index] += grads[z_offset + out_row + out_col0 + kcol];
							}
						}
					}
				}
			}
			grads += out.sp * input.c;
		}
	}
}

#ifdef GPU
void forward_upsample_gpu(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	float* Z = l->gpu.Z;
	int ksize = (int)k.ksize;
	int w = (int)in.w;
	int h = (int)in.h;
	size_t batch_size = net->batch_size;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t a = 0; a < l->n_inputs; a++) {
			layer_input input = l->inputs[a];
			launch_forward_upsample_kernel(&input.gpu.data[b * input.stride], Z, w, h, (int)input.c, ksize, 1);
			Z += out.sp * input.c;
		}
	}
	if (l->activation) l->activate(l->gpu.Z, l->gpu.output, out.n, batch_size);
	if (net->training) zero_array_gpu(l->gpu.grads, (int)(out.n * batch_size));
}

void backward_upsample_gpu(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	kernel_params k = l->k;
	int batch_size = (int)net->batch_size;
	float* grads = l->gpu.grads;
	if (l->activation) get_activation_grads_gpu(l, batch_size);
	int ksize = (int)k.ksize;
	int w = (int)in.w;
	int h = (int)in.h;
	int out_w = (int)out.w;
	int out_wh = out_w * (int)out.h;
	for (int b = 0; b < batch_size; b++) {
		for (int a = 0; a < l->n_inputs; a++) {
			layer_input input = l->inputs[a];
			launch_backward_upsample_kernel(&input.gpu.grads[b * input.stride], grads, w, h, (int)input.c, ksize, 1);
			grads += out_wh * input.c;
		}
	}
}
#else
#pragma warning (suppress:4100)
void forward_upsample_gpu(layer* l, network* net) {
	gpu_not_defined();
}
#pragma warning (suppress:4100)
void backward_upsample_gpu(layer* l, network* net) {
	gpu_not_defined();
}
#endif

/*** TESTING ***/

//void test_forward_upsample(void) {
//	layer l = { 0 };
//	network net = { 0 };
//	net.batch_size = 2;
//	l.w = 2;
//	l.h = 2;
//	l.ksize = 2;
//
//	layer inl1 = { 0 };
//	layer inl2 = { 0 };
//	inl1.out_w = l.w;
//	inl2.out_w = l.w;
//	inl1.out_h = l.h;
//	inl2.out_h = l.h;
//	inl1.out_c = 1;
//	inl2.out_c = 2;
//	inl1.out_n = inl1.out_w * inl1.out_h * inl1.out_c;
//	inl2.out_n = inl2.out_w * inl2.out_h * inl2.out_c;
//	inl1.output = (float*)xcalloc(inl1.out_n * net.batch_size, sizeof(float));
//	fill_array_increment(inl1.output, inl1.out_n * net.batch_size, 1.0F, 2.0F);
//	inl2.output = (float*)xcalloc(inl2.out_n * net.batch_size, sizeof(float));
//	fill_array_increment(inl2.output, inl2.out_n * net.batch_size, 0.5F, 1.0F);
//
//	l.c = inl1.out_c + inl2.out_c;
//	l.n = l.w * l.h * l.c;
//	l.out_w = l.w * l.ksize;
//	l.out_h = l.h * l.ksize;
//	l.out_c = l.c;
//	l.out_n = l.out_w * l.out_h * l.out_c;
//	l.output = (float*)xcalloc(l.out_n * net.batch_size, sizeof(float));
//	l.Z = l.output;
//
//	l.in_layers = (layer**)xcalloc(2, sizeof(layer*));
//	l.in_layers[0] = &inl1;
//	l.in_layers[1] = &inl2;
//	l.in_ids.n = 2;
//	forward_upsample(&l, &net);
//
//	pprint_mat_batch(inl1.output, inl1.out_w, inl1.out_h, inl1.out_c, net.batch_size);
//	pprint_mat_batch(inl2.output, inl2.out_w, inl2.out_h, inl2.out_c, net.batch_size);
//	pprint_mat_batch(l.output, l.out_w, l.out_h, l.out_c, net.batch_size);
//}
//
//void test_backward_upsample(void) {
//	layer l = { 0 };
//	network net = { 0 };
//	net.batch_size = 2;
//	l.w = 4;
//	l.h = 4;
//	l.ksize = 2;
//
//	layer inl1 = { 0 };
//	layer inl2 = { 0 };
//	inl1.out_w = l.w;
//	inl2.out_w = l.w;
//	inl1.out_h = l.h;
//	inl2.out_h = l.h;
//	inl1.out_c = 1;
//	inl2.out_c = 2;
//	inl1.out_n = inl1.out_w * inl1.out_h * inl1.out_c;
//	inl2.out_n = inl2.out_w * inl2.out_h * inl2.out_c;
//	inl1.grads = (float*)xcalloc(inl1.out_n * net.batch_size, sizeof(float));
//	inl2.grads = (float*)xcalloc(inl2.out_n * net.batch_size, sizeof(float));
//
//	l.c = inl1.out_c + inl2.out_c;
//	l.n = l.w * l.h * l.c;
//	l.out_w = l.w * l.ksize;
//	l.out_h = l.h * l.ksize;
//	l.out_c = l.c;
//	l.out_n = l.out_w * l.out_h * l.out_c;
//	l.grads = (float*)xcalloc(l.out_n * net.batch_size, sizeof(float));
//
//	fill_array_increment(l.grads, l.out_n * net.batch_size, 1.0F, 1.0F);
//	l.in_layers = (layer**)xcalloc(2, sizeof(layer*));
//	l.in_layers[0] = &inl1;
//	l.in_layers[1] = &inl2;
//	l.in_ids.n = 2;
//	backward_upsample(&l, &net);
//
//
//	pprint_mat_batch(l.grads, l.out_w, l.out_h, l.out_c, net.batch_size);
//	pprint_mat_batch(inl1.grads, inl1.out_w, inl1.out_h, inl1.out_c, net.batch_size);
//	pprint_mat_batch(inl2.grads, inl2.out_w, inl2.out_h, inl2.out_c, net.batch_size);
//}