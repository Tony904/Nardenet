#include "layer_avgpool_global.h"
#include <omp.h>
#include "derivatives.h"
#include "utils.h"
#include "xallocs.h"
#include "xcuda.h"
#include "blas.h"


void forward_avgpool_global(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	float* Z = l->Z;
	size_t batch_size = net->batch_size;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t a = 0; a < l->n_inputs; a++) {
			layer_input input = l->inputs[a];
			size_t b_offset = b * input.stride;
			size_t ch;
#pragma omp parallel for
			for (ch = 0; ch < input.c; ch++) {
				float sum = 0.0F;
				size_t offset = ch * in.sp + b_offset;
				for (size_t s = 0; s < in.sp; s++) {
					sum += input.data[offset + s];
				}
				Z[ch] = sum / in.sp;
			}
			Z += input.c;
		}
	}
	if (l->activation) l->activate(l->Z, l->output, out.n, net->batch_size);
	if (net->training) zero_array(l->grads, out.n * batch_size);
}

void backward_avgpool_global(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->grads;
	if (l->activation) get_activation_grads(l, batch_size);
	tensor_dims in = l->in;
	float divisor = (float)in.sp;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t a = 0; a < l->n_inputs; a++) {
			layer_input input = l->inputs[a];
			size_t b_offset = b * input.stride;
			size_t ch;
#pragma omp parallel for
			for (ch = 0; ch < input.c; ch++) {
				size_t offset = ch * in.sp + b_offset;
				float grad = grads[ch] / divisor;
				for (size_t s = 0; s < in.sp; s++) {
					input.grads[offset + s] += grad;
				}
			}
			grads += input.c;
		}
	}
}

#ifdef GPU
void forward_avgpool_global_gpu(layer* l, network* net) {
	tensor_dims in = l->in;
	tensor_dims out = l->out;
	float* Z = l->gpu.Z;
	size_t batch_size = net->batch_size;
	if (l->n_inputs > 1) {
		for (size_t b = 0; b < batch_size; b++) {
			for (size_t a = 0; a < l->n_inputs; a++) {
				layer_input input = l->inputs[a];
				launch_forward_avgpool_global_kernel(&input.gpu.data[b * input.stride], Z, (int)in.sp, (int)input.c, 1);
				Z += input.c;
			}
		}
	}
	else launch_forward_avgpool_global_kernel(l->inputs[0].gpu.data, Z, (int)in.sp, (int)in.c, (int)batch_size);
	if (l->activation) l->activate(l->gpu.Z, l->gpu.output, out.n, batch_size);
	if (net->training) zero_array_gpu(l->gpu.grads, (int)(out.n * batch_size));
}

void backward_avgpool_global_gpu(layer* l, network* net) {
	tensor_dims in = l->in;
	int batch_size = (int)net->batch_size;
	float* grads = l->gpu.grads;
	if (l->activation) get_activation_grads_gpu(l, batch_size);
	if (l->n_inputs > 1) {
		for (size_t b = 0; b < batch_size; b++) {
			for (size_t a = 0; a < l->n_inputs; a++) {
				layer_input input = l->inputs[a];
				launch_backward_avgpool_global_kernel(&input.gpu.grads[b * input.stride], grads, (int)in.sp, (int)input.c, 1);
				grads += input.c;
			}
		}
	}
	else launch_backward_avgpool_global_kernel(l->inputs[0].gpu.grads, grads, (int)in.sp, (int)in.c, batch_size);
}
#else
#pragma warning (suppress:4100)
void forward_avgpool_global_gpu(layer* l, network* net) {
	gpu_not_defined();
}
#pragma warning (suppress:4100)
void backward_avgpool_global_gpu(layer* l, network* net) {
	gpu_not_defined();
}
#endif

/*** TESTING ***/

//void test_forward_avgpool_global(void) {
//	layer l = { 0 };
//	network net = { 0 };
//	net.batch_size = 2;
//	l.w = 2;
//	l.h = 2;
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
//	//fill_array_rand_float(inl1.output, inl1.out_n * net.batch_size, 0.0, 1.0);
//	fill_array_increment(inl1.output, inl1.out_n * net.batch_size, 0.0F, 2.0F);
//	inl2.output = (float*)xcalloc(inl2.out_n * net.batch_size, sizeof(float));
//	//fill_array_rand_float(inl2.output, inl2.out_n * net.batch_size, 0.0, 1.0);
//	fill_array_increment(inl2.output, inl2.out_n * net.batch_size, 0.0F, 1.0F);
//
//	l.c = inl1.out_c + inl2.out_c;
//	l.n = l.w * l.h * l.c;
//	l.out_w = 1;
//	l.out_h = 1;
//	l.out_c = l.c;
//	l.out_n = l.out_w * l.out_h * l.out_c;
//	l.output = (float*)xcalloc(l.out_n * net.batch_size, sizeof(float));
//	l.Z = l.output;
//
//	inl1.grads = (float*)xcalloc(inl1.out_n * net.batch_size, sizeof(float));
//	inl2.grads = (float*)xcalloc(inl2.out_n * net.batch_size, sizeof(float));
//	l.in_layers = (layer**)xcalloc(2, sizeof(layer*));
//	l.in_layers[0] = &inl1;
//	l.in_layers[1] = &inl2;
//	l.in_ids.n = 2;
//	forward_avgpool_global(&l, &net);
//
//	pprint_mat(inl1.output, (int)inl1.out_w, (int)inl1.out_h, (int)inl1.out_c * (int)net.batch_size);
//	pprint_mat(inl2.output, (int)inl2.out_w, (int)inl2.out_h, (int)inl2.out_c * (int)net.batch_size);
//	pprint_mat(l.output, (int)l.out_w, (int)l.out_h, (int)l.out_c * (int)net.batch_size);
//}
//
//void test_backward_avgpool_global(void) {
//	layer l = { 0 };
//	network net = { 0 };
//	net.batch_size = 2;
//	l.w = 2;
//	l.h = 2;
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
//	//fill_array_rand_float(inl1.output, inl1.out_n * net.batch_size, 0.0, 1.0);
//	fill_array_increment(inl1.output, inl1.out_n * net.batch_size, 0.0F, 2.0F);
//	inl2.output = (float*)xcalloc(inl2.out_n * net.batch_size, sizeof(float));
//	//fill_array_rand_float(inl2.output, inl2.out_n * net.batch_size, 0.0, 1.0);
//	fill_array_increment(inl2.output, inl2.out_n * net.batch_size, 0.0F, 1.0F);
//
//	l.c = inl1.out_c + inl2.out_c;
//	l.n = l.w * l.h * l.c;
//	l.out_w = 1;
//	l.out_h = 1;
//	l.out_c = l.c;
//	l.out_n = l.out_w * l.out_h * l.out_c;
//	l.output = (float*)xcalloc(l.out_n * net.batch_size, sizeof(float));
//	l.Z = l.output;
//
//	inl1.grads = (float*)xcalloc(inl1.out_n * net.batch_size, sizeof(float));
//	inl2.grads = (float*)xcalloc(inl2.out_n * net.batch_size, sizeof(float));
//	l.in_layers = (layer**)xcalloc(2, sizeof(layer*));
//	l.in_layers[0] = &inl1;
//	l.in_layers[1] = &inl2;
//	l.in_ids.n = 2;
//	forward_avgpool_global(&l, &net);
//
//	l.grads = (float*)xcalloc(l.out_n * net.batch_size, sizeof(float));
//
//	fill_array_increment(l.grads, l.out_n * net.batch_size, 0.0F, 4.0F);
//	backward_avgpool_global(&l, &net);
//	pprint_mat(l.grads, (int)l.out_w, (int)l.out_h, (int)(l.out_c * net.batch_size));
//	pprint_mat(inl1.grads, (int)inl1.out_w, (int)inl1.out_h, (int)(inl1.out_c * net.batch_size));
//	pprint_mat(inl2.grads, (int)inl2.out_w, (int)inl2.out_h, (int)(inl2.out_c * net.batch_size));
//}