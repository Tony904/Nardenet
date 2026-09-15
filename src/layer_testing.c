#include "layer_testing.h"
#include "network.h"
#include "xallocs.h"
#include "blas.h"
#include "utils.h"
#include "layer_avgpool_local.h"
#include "xcuda.h"


#define MAX_INPUT_LAYERS 10


layer* new_test_pool_layer(size_t ksize, size_t pad, size_t stride, size_t w, size_t h, size_t c, size_t batch_size) {
	layer* l = (layer*)xcalloc(1, sizeof(layer));
	l->in_layers = (layer**)xcalloc(MAX_INPUT_LAYERS, sizeof(layer*));
	l->ksize = ksize;
	l->pad = pad;
	l->stride = stride;
	l->w = w;
	l->h = h;
	l->c = c;
	l->n = l->w * l->h * l->c;
	l->out_w = (l->w + 2 * l->pad - l->ksize) / l->stride + 1;
	l->out_h = (l->h + 2 * l->pad - l->ksize) / l->stride + 1;
	l->out_c = l->c;
	l->out_n = l->n;
	l->Z = (float*)xcalloc(l->n * batch_size, sizeof(float));
	l->act_inputs = l->Z;
	l->output = l->Z;
	return l;
}

layer* new_test_input_layer(size_t out_w, size_t out_h, size_t out_c, size_t batch_size) {
	layer* inl1 = (layer*)xcalloc(1, sizeof(layer));
	inl1->out_w = out_w;
	inl1->out_h = out_h;
	inl1->out_c = out_c;
	inl1->out_n = inl1->out_w * inl1->out_h * inl1->out_c;
	inl1->output = (float*)xcalloc(inl1->out_n * batch_size, sizeof(float));
	return inl1;
}

void add_test_input_layer(layer* l, size_t inl_out_w, size_t inl_out_h, size_t inl_out_c, size_t batch_size) {
	layer* inl = new_test_input_layer(inl_out_w, inl_out_h, inl_out_c, batch_size);
	if (l->in_ids.n >= MAX_INPUT_LAYERS) {
		printf("Failed to add input layer. Max input layers = %d\n", MAX_INPUT_LAYERS);
		return;
	}
	l->in_layers[l->in_ids.n] = inl;
	l->in_ids.n++;
}

void validate_layer_dimensions(layer* l) {
	size_t c = 0;
	for (size_t i = 0; i < l->in_ids.n; i++) {
		c += l->in_layers[i]->out_c;
	}
	if (l->c && c == l->c) {
		return;
	}
	printf("Invalid channels. Input channels = %zu, layer channels = %zu\n", c, l->c);
	wait_for_key_then_exit();
}

/* TEST FUNCTIONS */
void test_forward_avgpool_local(void) {
	network* net = (network*)xcalloc(1, sizeof(network));
	net->batch_size = 2;
	size_t w = 3;
	size_t h = 3;
	size_t c = 3;
	layer* l = new_test_pool_layer(2, 0, 1, w, h, c, net->batch_size);
	// Input layer 1
	add_test_input_layer(l, l->w, l->h, 1, net->batch_size);
	layer* inl1 = l->in_layers[0];
	fill_array_increment(inl1->output, inl1->out_n * net->batch_size, 0, 1);
	pprint_mat_batch(inl1->output, inl1->out_w, inl1->out_h, inl1->out_c, net->batch_size);
	// Input layer 2
	add_test_input_layer(l, l->w, l->h, 2, net->batch_size);
	layer* inl2 = l->in_layers[1];
	fill_array_increment(inl2->output, inl2->out_n * net->batch_size, 3, 1);
	pprint_mat_batch(inl2->output, inl2->out_w, inl2->out_h, inl2->out_c, net->batch_size);

	validate_layer_dimensions(l);

	forward_avgpool_local(l, net);

	pprint_mat_batch(l->output, l->out_w, l->out_h, l->out_c, net->batch_size);
}


void test_launch_forward_maxpool_standard_even_spatial_kernel(void) {
	network* net = (network*)xcalloc(1, sizeof(network));
	net->batch_size = 1;
	size_t w = 2;
	size_t h = 2;
	size_t c = 1;
	size_t stride = 2;
	size_t ksize = 2;
	layer* l = new_test_pool_layer(ksize, 0, stride, w, h, c, net->batch_size);
	size_t dst_size = l->n * l->c * net->batch_size;
	float* d_dst = 0;
	CHECK_CUDA(cudaMalloc(&d_dst, dst_size * sizeof(float)));
	float* d_grads = 0;
	CHECK_CUDA(cudaMalloc(&d_grads, dst_size * sizeof(float)));
	float** d_addresses = 0;
	CHECK_CUDA(cudaMalloc((void**)&d_addresses, dst_size * sizeof(float*)));
	
	// Input layer 1
	add_test_input_layer(l, l->w * stride, l->h * stride, 1, net->batch_size);
	layer* inl1 = l->in_layers[0];
	size_t src_size = inl1->out_n * net->batch_size;
	//fill_array_increment(inl1->output, src_size, 0, 1);
	fill_array_rand_float(inl1->output, src_size, 0, 5);
	pprint_mat_batch(inl1->output, inl1->out_w, inl1->out_h, inl1->out_c, net->batch_size);
	
	float* d_src = 0;
	CHECK_CUDA(cudaMalloc(&d_src, src_size * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(d_src, inl1->output, src_size * sizeof(float), cudaMemcpyHostToDevice));

	// Input layer 2
	/*add_test_input_layer(l, l->w, l->h, 2, net->batch_size);
	layer* inl2 = l->in_layers[1];
	fill_array_increment(inl2->output, inl2->out_n * net->batch_size, 3, 1);
	pprint_mat_batch(inl2->output, inl2->out_w, inl2->out_h, inl2->out_c, net->batch_size);*/

	validate_layer_dimensions(l);

	launch_forward_maxpool_standard_even_spatial_kernel(d_src, d_dst, d_grads, d_addresses,
		(int)inl1->out_w, (int)inl1->out_h, (int)l->w, (int)l->h, (int)l->n);
	/*launch_forward_maxpool_general_kernel(d_src, d_dst, d_grads, d_addresses,
		(int)inl1->out_w, (int)inl1->out_h, (int)l->w, (int)l->h, (int)l->n, (int)ksize, (int)stride);*/

	print_gpu_float_array(d_dst, dst_size, "maxpool results");
}