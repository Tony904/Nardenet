#include "network.h"
#include <assert.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include "utils.h"
#include "xallocs.h"
#include "layer_conv.h"
#include "layer_classify.h"
#include "layer_dense.h"
#include "layer_maxpool.h"
#include "layer_residual.h"
#include "layer_detect.h"
#include "layer_avgpool_global.h"
#include "layer_avgpool_local.h"
#include "layer_upsample.h"
#include "layer_route.h"
#include "activations.h"
#include "loss.h"
#include "derivatives.h"
#include "data.h"
#include "blas.h"
#include "xcuda.h"
#include "xopencv.h"

#pragma warning (disable:4702)

void build_network_input_layer(network* net);
void build_layer(layer* l, network* net);
void build_conv_layer(layer* l, network* net);
void build_dense_layer(layer* l, network* net);
void build_maxpool_layer(layer* l, network* net);
void build_avgpool_global_layer(layer* l, network* net);
void build_avgpool_local_layer(layer* l, network* net);
void build_residual_layer(layer* l, network* net);
void build_upsample_layer(layer* l, network* net);
void build_classify_layer(layer* l, network* net);
void build_detect_layer(layer* l, network* net);

void set_activate(layer* l, int use_gpu);
void set_loss(layer* l, int use_gpu);
void set_regularization(network* net);

void backward_none(layer* l, network* net);
void update_none(layer* l, network* net);

size_t get_train_params_count(network* net);
size_t get_layer_param_count(layer* l);

void free_network(network* net);
void free_network_layers(network* net);
void free_layer_fields(layer* l, network* net);

void print_layer_conv(layer* l);
void print_layer_classify(layer* l);
void print_layer_maxpool(layer* l);
void print_layer_residual(layer* l);

void get_layer_type_str(char* buf, size_t bufsize, LAYER_TYPE lt);
void get_input_ids_str(char* buf, size_t bufsize, layer_input* inputs, size_t n_inputs);


network* new_network(size_t num_of_layers) {
	network* net = (network*)xcalloc(1, sizeof(network));
	net->input = (layer*)xcalloc(1, sizeof(layer));
	net->n_layers = num_of_layers;
	net->layers = (layer*)xcalloc(num_of_layers, sizeof(layer));
	return net;
}

void build_network(network* net) {
	if (net->type == NET_DETECT) {
		net->data.detector.current_batch = (det_sample**)xcalloc(net->batch_size, sizeof(det_sample*));
		net->anchors = (bbox*)xcalloc(net->n_anchors, sizeof(bbox));
	}
	build_network_input_layer(net);
	size_t largest_workspace = 0;
	for (int i = 0; i < net->n_layers; i++) {
		layer* l = &net->layers[i];
		build_layer(l, net);
		if (l->type == LAYER_CONV) {
			size_t ws_size = l->out.w * l->out.h * l->k.ksize * l->k.ksize * l->in.c;
			largest_workspace = max(largest_workspace, ws_size);
		}
	}
	if (largest_workspace > 0) {
		net->workspace = (float*)xcalloc(largest_workspace, sizeof(float));
		if (net->use_gpu) {
			CUDA_MALLOC(&net->gpu.workspace, largest_workspace, sizeof(float));
		}
	}
	net->workspace_size = largest_workspace;
	net->current_learning_rate = net->learning_rate;
	set_regularization(net);
}

void build_layer(layer* l, network* net) {
	assert(l->type != LAYER_NONE);
	if (l->type == LAYER_CONV) build_conv_layer(l, net);
	else if (l->type == LAYER_CLASSIFY) build_classify_layer(l, net);
	else if (l->type == LAYER_MAXPOOL) build_maxpool_layer(l, net);
	else if (l->type == LAYER_DENSE) build_dense_layer(l, net);
	else if (l->type == LAYER_RESIDUAL) build_residual_layer(l, net);
	else if (l->type == LAYER_DETECT) build_detect_layer(l, net);
	else if (l->type == LAYER_AVGPOOL_GLOBAL) build_avgpool_global_layer(l, net);
	else if (l->type == LAYER_UPSAMPLE) build_upsample_layer(l, net);
	else {
		printf("Unknown layer type: %d\n", (int)l->type);
		wait_for_key_then_exit();
	}
}

void build_network_input_layer(network* net) {
	layer* l = net->input;
	l->type = LAYER_NETWORK_INPUT;
	l->train = 0;
	l->id = -1;
	l->in.w = net->w;
	l->in.h = net->h;
	l->in.c = net->c;
	l->in.n = l->in.w * l->in.h * l->in.c;
	l->out.w = l->in.w;
	l->out.h = l->in.h;
	l->out.c = l->in.c;
	l->out.n = l->in.n;
	l->output = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.output, l->out.n * net->batch_size, sizeof(float));
	}
}

void set_layer_inputs(layer* l, network* net) {
	layer* ls = net->layers;
	for (size_t i = 0; i < l->n_inputs; i++) {
		layer_input* input = &l->inputs[i];
		if (input->n_chunks == 0) {
			assert(input->chunk_idx == 0);
			input->n_chunks = 1;
		}
		layer* src;
		if (l->id == 0) {
			assert(l->n_inputs == 1);
			src = net->input;
		}
		else {
			src = &ls[input->source_id];
		}
		input->source = src;
		if (src->out.c % input->n_chunks != 0) {
			printf("Cannot evenly split %zu channels into %zu chunks. Layer ID: %d\n", src->out.c, input->n_chunks, l->id);
			wait_for_key_then_exit();
		}
		input->c = src->out.c / input->n_chunks;
		input->n = input->c * src->out.w * src->out.h;
		input->stride = src->out.n;
		input->data = &src->output[input->chunk_idx * input->n];
		input->grads = l->id ?  &src->grads[input->chunk_idx * input->n] : 0;
		if (net->use_gpu) {
			input->gpu.data = &src->gpu.output[input->chunk_idx * input->n];
			input->gpu.grads = l->id ? &src->gpu.grads[input->chunk_idx * input->n] : 0;
		}
	}
}

void set_input_dimensions(layer* l) {
	l->in.w = l->inputs[0].source->out.w;
	l->in.h = l->inputs[0].source->out.h;
	l->in.c = l->inputs[0].c;
	for (size_t i = 1; i < l->n_inputs; i++) {
		layer_input* input = &l->inputs[i];
		if (l->in.w != input->source->out.w || l->in.h != input->source->out.h) {
			printf("Invalid input layer dimensions. Width and height must match.\n"
				"Layer %d %zux%zu, Input layer %d %zux%zu\n",
				l->id, l->in.w, l->in.h, input->source_id, input->source->out.w, input->source->out.h);
			wait_for_key_then_exit();
		}
		if (l->type == LAYER_RESIDUAL) {
			if (input->c != l->in.c) {
				printf("Inputs to residual layers must have equal depth.\nResidual layer ID %d, Input layer ID %d\n", l->id, input->source_id);
				printf("Residual layer channels: %zu | Input layer channels: %zu\n", l->in.c, input->c);
				wait_for_key_then_exit();
			}
		}
		else {
			l->in.c += input->c;
		}
	}
	l->in.n = l->in.w * l->in.h * l->in.c;
	l->in.sp = l->in.w * l->in.h;
}

void build_conv_layer(layer* l, network* net) {
	if (!l->k.n_groups) l->k.n_groups = 1;
	if (!l->k.stride) l->k.stride = 1;
	if (!l->k.n_filters) {
		printf("Error: n_filters cannot be zero. (layer %d)\n", l->id);
		wait_for_key_then_exit();
	}
	if (l->k.n_filters % l->k.n_groups > 0) {
		printf("Cannot evenly distribute %zu filters between %zu groups. (layer %d)\n", l->k.n_filters, l->k.n_groups, l->id);
		wait_for_key_then_exit();
	}

	set_layer_inputs(l, net);
	set_input_dimensions(l);

	if (l->in.c % l->k.n_groups > 0) {
		printf("Cannot evenly distribute %zu channels between %zu groups. (layer %d)\n", l->in.c, l->k.n_groups, l->id);
		wait_for_key_then_exit();
	}

	// Calculate output dimensions.
	l->out.w = ((l->in.w + (l->k.pad * 2) - l->k.ksize) / l->k.stride) + 1;
	l->out.h = ((l->in.h + (l->k.pad * 2) - l->k.ksize) / l->k.stride) + 1;
	l->out.c = l->k.n_filters;
	l->out.n = l->out.w * l->out.h * l->out.c;

	l->Z = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
	l->n_weights = l->k.n_filters * l->k.ksize * l->k.ksize * (l->in.c / l->k.n_groups);
	l->weights = (float*)xcalloc(l->n_weights, sizeof(float));
	l->biases = (float*)xcalloc(l->k.n_filters, sizeof(float));
	l->grads = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
	l->weight_grads = (float*)xcalloc(l->n_weights, sizeof(float));
	l->bias_grads = (float*)xcalloc(l->k.n_filters, sizeof(float));
	l->weight_velocities = (float*)xcalloc(l->n_weights, sizeof(float));
	l->bias_velocities = (float*)xcalloc(l->k.n_filters, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.Z, l->out.n * net->batch_size, sizeof(float));
		CUDA_MALLOC(&l->gpu.weights, l->n_weights, sizeof(float));
		CUDA_MALLOC(&l->gpu.biases, l->k.n_filters, sizeof(float));
		CUDA_MALLOC(&l->gpu.grads, l->out.n * net->batch_size, sizeof(float));
		CUDA_MALLOC(&l->gpu.weight_grads, l->n_weights, sizeof(float));
		CUDA_MALLOC(&l->gpu.bias_grads, l->k.n_filters, sizeof(float));
		CUDA_MALLOC(&l->gpu.weight_velocities, l->n_weights, sizeof(float));
		CUDA_MALLOC(&l->gpu.bias_velocities, l->k.n_filters, sizeof(float));
	}
	if (l->batchnorm) {
		l->Z_norm = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
		l->act_inputs = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
		l->means = (float*)xcalloc(l->k.n_filters, sizeof(float));
		l->variances = (float*)xcalloc(l->k.n_filters, sizeof(float));
		l->gammas = (float*)xcalloc(l->k.n_filters, sizeof(float));
		fill_array(l->gammas, l->k.n_filters, 1.0F);
		l->gamma_grads = (float*)xcalloc(l->k.n_filters, sizeof(float));
		l->gamma_velocities = (float*)xcalloc(l->k.n_filters, sizeof(float));
		l->rolling_means = (float*)xcalloc(l->k.n_filters, sizeof(float));
		l->rolling_variances = (float*)xcalloc(l->k.n_filters, sizeof(float));
		if (net->use_gpu) {
			CUDA_MALLOC(&l->gpu.Z_norm, l->out.n * net->batch_size, sizeof(float));
			CUDA_MALLOC(&l->gpu.act_inputs, l->out.n * net->batch_size, sizeof(float));
			CUDA_MALLOC(&l->gpu.means, l->k.n_filters, sizeof(float));
			CUDA_MALLOC(&l->gpu.variances, l->k.n_filters, sizeof(float));
			CUDA_MALLOC(&l->gpu.gammas, l->k.n_filters, sizeof(float));
			CUDA_MEMCPY_H2D(l->gpu.gammas, l->gammas, l->k.n_filters * sizeof(float));
			CUDA_MALLOC(&l->gpu.gamma_grads, l->k.n_filters, sizeof(float));
			CUDA_MALLOC(&l->gpu.gamma_velocities, l->k.n_filters, sizeof(float));
			CUDA_MALLOC(&l->gpu.rolling_means, l->k.n_filters, sizeof(float));
			CUDA_MALLOC(&l->gpu.rolling_variances, l->k.n_filters, sizeof(float));
		}
	}
	else {
		l->act_inputs = l->Z;
		l->gpu.act_inputs = l->gpu.Z;
	}
	if (l->activation) {
		l->output = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
		if (net->use_gpu) {
			CUDA_MALLOC(&l->gpu.output, l->out.n * net->batch_size, sizeof(float));
		}
	}
	else {
		l->output = l->act_inputs;
		l->gpu.output = l->gpu.act_inputs;
	}
	
	if (net->use_gpu == 2) {
		l->forward = forward_conv_cpu_gpu_compare;
		l->backward = backward_conv_cpu_gpu_compare;
		l->update = update_conv_cpu_gpu_compare;
	}
	else if (net->use_gpu == 1) {
		l->forward = forward_conv_gpu;
		l->backward = backward_conv_gpu;
		l->update = update_conv_gpu;
	}
	else {
		l->forward = forward_conv;
		l->backward = backward_conv;
		l->update = update_conv;
	}
	
	set_activate(l, net->use_gpu);
}

void build_dense_layer(layer* l, network* net) {
	set_layer_inputs(l, net);
	set_input_dimensions(l);

	l->out.w = 1;
	l->out.h = 1;
	l->out.c = l->k.n_filters;
	l->out.n = l->out.w * l->out.h * l->out.c;

	l->Z = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
	l->n_weights = l->in.w * l->in.h * l->in.c * l->k.n_filters;
	l->weights = (float*)xcalloc(l->n_weights, sizeof(float));
	l->biases = (float*)xcalloc(l->k.n_filters, sizeof(float));
	l->grads = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
	l->weight_grads = (float*)xcalloc(l->n_weights, sizeof(float));
	l->bias_grads = (float*)xcalloc(l->k.n_filters, sizeof(float));
	l->weight_velocities = (float*)xcalloc(l->n_weights, sizeof(float));
	l->bias_velocities = (float*)xcalloc(l->k.n_filters, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.Z, l->out.n * net->batch_size, sizeof(float));
		CUDA_MALLOC(&l->gpu.weights, l->n_weights, sizeof(float));
		CUDA_MALLOC(&l->gpu.biases, l->k.n_filters, sizeof(float));
		CUDA_MALLOC(&l->gpu.grads, l->out.n * net->batch_size, sizeof(float));
		CUDA_MALLOC(&l->gpu.weight_grads, l->n_weights, sizeof(float));
		CUDA_MALLOC(&l->gpu.bias_grads, l->k.n_filters, sizeof(float));
		CUDA_MALLOC(&l->gpu.weight_velocities, l->n_weights, sizeof(float));
		CUDA_MALLOC(&l->gpu.bias_velocities, l->k.n_filters, sizeof(float));
	}
	if (l->batchnorm) {
		l->Z_norm = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
		l->act_inputs = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
		l->means = (float*)xcalloc(l->k.n_filters, sizeof(float));
		l->variances = (float*)xcalloc(l->k.n_filters, sizeof(float));
		l->gammas = (float*)xcalloc(l->k.n_filters, sizeof(float));
		fill_array(l->gammas, l->k.n_filters, 1.0F);
		l->gamma_grads = (float*)xcalloc(l->k.n_filters, sizeof(float));
		l->gamma_velocities = (float*)xcalloc(l->k.n_filters, sizeof(float));
		l->rolling_means = (float*)xcalloc(l->k.n_filters, sizeof(float));
		l->rolling_variances = (float*)xcalloc(l->k.n_filters, sizeof(float));
		if (net->use_gpu) {
			CUDA_MALLOC(&l->gpu.Z_norm, l->out.n * net->batch_size, sizeof(float));
			CUDA_MALLOC(&l->gpu.act_inputs, l->out.n * net->batch_size, sizeof(float));
			CUDA_MALLOC(&l->gpu.means, l->k.n_filters, sizeof(float));
			CUDA_MALLOC(&l->gpu.variances, l->k.n_filters, sizeof(float));
			CUDA_MALLOC(&l->gpu.gammas, l->k.n_filters, sizeof(float));
			CUDA_MEMCPY_H2D(l->gpu.gammas, l->gammas, l->k.n_filters * sizeof(float));
			CUDA_MALLOC(&l->gpu.gamma_grads, l->k.n_filters, sizeof(float));
			CUDA_MALLOC(&l->gpu.gamma_velocities, l->k.n_filters, sizeof(float));
			CUDA_MALLOC(&l->gpu.rolling_means, l->k.n_filters, sizeof(float));
			CUDA_MALLOC(&l->gpu.rolling_variances, l->k.n_filters, sizeof(float));
		}
	}
	else {
		l->act_inputs = l->Z;
		l->gpu.act_inputs = l->gpu.Z;
	}

	if (l->activation) {
		l->output = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
		if (net->use_gpu) {
			CUDA_MALLOC(&l->gpu.output, l->out.n * net->batch_size, sizeof(float));
		}
	}
	else {
		l->output = l->act_inputs;
		l->gpu.output = l->gpu.act_inputs;
	}

	if (net->use_gpu == 2) {
		l->forward = forward_dense_cpu_gpu_compare;
		l->backward = backward_dense_cpu_gpu_compare;
		l->update = update_conv_cpu_gpu_compare;
	}
	else if (net->use_gpu == 1) {
		l->forward = forward_dense_gpu;
		l->backward = backward_dense_gpu;
		l->update = update_conv_gpu;
	}
	else {
		l->forward = forward_dense;
		l->backward = backward_dense;
		l->update = update_conv;
	}

	set_activate(l, net->use_gpu);
}

void build_maxpool_layer(layer* l, network* net) {
	set_layer_inputs(l, net);
	set_input_dimensions(l);

	l->out.w = ((l->in.w - l->k.ksize) / l->k.stride) + 1;
	l->out.h = ((l->in.h - l->k.ksize) / l->k.stride) + 1;
	l->out.c = l->in.c;
	l->out.n = l->out.w * l->out.h * l->out.c;

	l->output = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
	l->grads = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
	l->maxpool_addresses = (float**)xcalloc(l->out.n * net->batch_size, sizeof(float*));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.output, l->out.n * net->batch_size, sizeof(float));
		CUDA_MALLOC(&l->gpu.grads, l->out.n * net->batch_size, sizeof(float));
		CUDA_MALLOC((void**)(&l->gpu.maxpool_addresses), l->out.n * net->batch_size, sizeof(float*));
	}

	if (net->use_gpu) {
		l->forward = forward_maxpool_gpu;
		l->backward = backward_maxpool_gpu;
	}
	else {
		//l->forward = (l->k.ksize != 2 || l->k.stride != 2) ? forward_maxpool_general : forward_maxpool;
		l->forward = forward_maxpool_general;
		l->backward = backward_maxpool;
	}
	l->update = update_none;
}

void build_residual_layer(layer* l, network* net) {
	set_layer_inputs(l, net);
	set_input_dimensions(l);

	l->out.w = l->in.w;
	l->out.h = l->in.h;
	l->out.c = l->in.c;
	l->out.n = l->out.w * l->out.h * l->out.c;

	l->Z = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
	l->act_inputs = l->Z;
	l->grads = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.Z, l->out.n * net->batch_size, sizeof(float));
		l->gpu.act_inputs = l->gpu.Z;
		CUDA_MALLOC(&l->gpu.grads, l->out.n * net->batch_size, sizeof(float));
	}

	if (l->activation) {
		l->output = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
		if (net->use_gpu) CUDA_MALLOC(&l->gpu.output, l->out.n * net->batch_size, sizeof(float));
	}
	else {
		l->output = l->Z;
		l->gpu.output = l->gpu.Z;
	}

	if (net->use_gpu) {
		l->forward = forward_residual_gpu;
		l->backward = backward_residual_gpu;
	}
	else {
		l->forward = forward_residual;
		l->backward = backward_residual;
	}
	l->update = update_none;
	
	set_activate(l, net->use_gpu);
}

void build_route_layer(layer* l, network* net) {
	set_layer_inputs(l, net);
	set_input_dimensions(l);

	l->out.w = l->in.w;
	l->out.h = l->in.h;
	l->out.c = l->in.c;
	l->out.n = l->out.w * l->out.h * l->out.c;

	l->Z = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
	l->act_inputs = l->Z;
	l->grads = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.Z, l->out.n * net->batch_size, sizeof(float));
		l->gpu.act_inputs = l->gpu.Z;
		CUDA_MALLOC(&l->gpu.grads, l->out.n * net->batch_size, sizeof(float));
	}

	if (l->activation) {
		l->output = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
		if (net->use_gpu) CUDA_MALLOC(&l->gpu.output, l->out.n * net->batch_size, sizeof(float));
	}
	else {
		l->output = l->Z;
		l->gpu.output = l->gpu.Z;
	}

	if (net->use_gpu) {
		l->forward = forward_route_gpu;
		l->backward = backward_route_gpu;
	}
	else {
		l->forward = forward_route;
		l->backward = backward_route;
	}
	l->update = update_none;
	
	set_activate(l, net->use_gpu);
}

void build_avgpool_global_layer(layer* l, network* net) {
	set_layer_inputs(l, net);
	set_input_dimensions(l);

	l->out.w = 1;
	l->out.h = 1;
	l->out.c = l->in.c;
	l->out.n = l->out.w * l->out.h * l->out.c;

	l->Z = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
	l->act_inputs = l->Z;
	l->grads = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.Z, l->out.n * net->batch_size, sizeof(float));
		l->gpu.act_inputs = l->gpu.Z;
		CUDA_MALLOC(&l->gpu.grads, l->out.n * net->batch_size, sizeof(float));
	}

	if (l->activation) {
		l->output = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
		if (net->use_gpu) {
			CUDA_MALLOC(&l->gpu.output, l->out.n * net->batch_size, sizeof(float));
		}
	}
	else {
		l->output = l->Z;
		l->gpu.output = l->gpu.Z;
	}

	if (net->use_gpu) {
		l->forward = forward_avgpool_global_gpu;
		l->backward = backward_avgpool_global_gpu;
	}
	else {
		l->forward = forward_avgpool_global;
		l->backward = backward_avgpool_global;
	}
	l->update = update_none;

	set_activate(l, net->use_gpu);
}

void build_avgpool_local_layer(layer* l, network* net) {
	set_layer_inputs(l, net);
	set_input_dimensions(l);

	l->out.w = ((l->in.w + (l->k.pad * 2) - l->k.ksize) / l->k.stride) + 1;
	l->out.h = ((l->in.h + (l->k.pad * 2) - l->k.ksize) / l->k.stride) + 1;
	l->out.c = l->in.c;
	l->out.n = l->out.w * l->out.h * l->out.c;

	l->Z = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
	l->act_inputs = l->Z;
	l->grads = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.Z, l->out.n * net->batch_size, sizeof(float));
		l->gpu.act_inputs = l->gpu.Z;
		CUDA_MALLOC(&l->gpu.grads, l->out.n * net->batch_size, sizeof(float));
	}

	if (l->activation) {
		l->output = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
		if (net->use_gpu) {
			CUDA_MALLOC(&l->gpu.output, l->out.n * net->batch_size, sizeof(float));
		}
	}
	else {
		l->output = l->Z;
		l->gpu.output = l->gpu.Z;
	}

	if (net->use_gpu) {
		l->forward = forward_avgpool_global_gpu;
		l->backward = backward_avgpool_global_gpu;
	}
	else {
		l->forward = forward_avgpool_global;
		l->backward = backward_avgpool_global;
	}
	l->update = update_none;

	set_activate(l, net->use_gpu);
}

void build_upsample_layer(layer* l, network* net) {
	set_layer_inputs(l, net);
	set_input_dimensions(l);

	l->out.w = l->in.w * l->k.ksize;
	l->out.h = l->in.h * l->k.ksize;
	l->out.c = l->in.c;
	l->out.n = l->out.w * l->out.h * l->out.c;
	l->Z = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.Z, l->out.n * net->batch_size, sizeof(float));
	}

	l->act_inputs = l->Z;
	l->gpu.act_inputs = l->gpu.Z;

	if (l->activation) {
		l->output = (float*)xcalloc(l->out.n * net->batch_size, sizeof(float));
		if (net->use_gpu) CUDA_MALLOC(&l->gpu.output, l->out.n * net->batch_size, sizeof(float));
	}
	else {
		l->output = l->act_inputs;
		l->gpu.output = l->gpu.act_inputs;
	}

	if (net->use_gpu) {
		l->forward = forward_upsample_gpu;
		l->backward = backward_upsample_gpu;
	}
	else {
		l->forward = forward_upsample;
		l->backward = backward_upsample;
	}
	l->update = update_none;

	set_activate(l, net->use_gpu);
}

// i = layer index in net->layers
void build_classify_layer(layer* l, network* net) {
	if (net->type == NET_NONE || net->type == NET_CLASSIFY) net->type = NET_CLASSIFY;
	else {
		printf("Invalid network cfg: Cannot have a classify layer in a non-classifier network.\n");
		wait_for_key_then_exit();
	}

	if (l->n_classes == 0) l->n_classes = net->n_classes;

	if (l->n_inputs > 1) {
		printf("Classify layers cannot have multiple input layers. (layer %d)\n", l->id);
	}
	set_layer_inputs(l, net);
	set_input_dimensions(l);

	if (l->in.w != 1 || l->in.h != 1 || l->in.c != l->n_classes) {
		printf("The layer outputting to a classify layer must have a width, height, and depth of 1x1x%zu. (is %zux%zux%zu)\n", l->n_classes, l->in.w, l->in.h, l->in.c);
		wait_for_key_then_exit();
	}

	l->output = l->inputs[0].source->output;
	l->grads = l->inputs[0].source->grads;
	l->gpu.output = l->inputs[0].source->gpu.output;
	l->gpu.grads = l->inputs[0].source->gpu.grads;

	l->truth = (float*)xcalloc(l->n_classes * net->batch_size, sizeof(float));
	l->errors = (float*)xcalloc(l->n_classes * net->batch_size, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.truth, l->n_classes * net->batch_size, sizeof(float));
		CUDA_MALLOC(&l->gpu.errors, l->n_classes * net->batch_size, sizeof(float));
		CUDA_MALLOC(&l->gpu.loss, 1, sizeof(float));
	}

	if (net->use_gpu == 2) l->forward = forward_classify_cpu_gpu_compare;
	else if (net->use_gpu == 1) l->forward = forward_classify_gpu;
	else l->forward = forward_classify;
	l->backward = backward_none;  // all backprop stuff is done in forward pass
	l->update = update_none;

	set_loss(l, net->use_gpu);
}

void build_detect_layer(layer* l, network* net) {
	if (net->type != NET_DETECT) {
		printf("Invalid network cfg: Cannot have a detect layer in a non-detector network.\n");
		wait_for_key_then_exit();
	}

	if (!l->n_classes) l->n_classes = net->n_classes;

	if (l->n_inputs > 1) {
		printf("Detect layers cannot have multiple input layers. (layer %d)\n", l->id);
	}
	set_layer_inputs(l, net);
	set_input_dimensions(l);

	if (l->inputs[0].source->activation != ACT_SIGMOID) {
		printf("Invalid cfg: The input layer to a detect layer must have an sigmoid activation.\n");
		wait_for_key_then_exit();
	}

	l->output = (float*)xcalloc(l->in.n * net->batch_size, sizeof(float));
	l->grads = (float*)xcalloc(l->in.n * net->batch_size, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.output, l->in.n * net->batch_size, sizeof(float));
		CUDA_MALLOC(&l->gpu.grads, l->in.n * net->batch_size, sizeof(float));
	}
	l->errors = (float*)xcalloc(net->batch_size, sizeof(float));  // only used for storing iou losses

	l->forward = forward_detect;
	if (net->use_gpu) l->backward = backward_detect_gpu;
	else l->backward = backward_detect;
	l->update = update_none;

	l->activation = ACT_NONE;

	// set anchors
	if (net->w != net->h || l->in.w != l->in.h) {
		printf("Network input and output width & height must be square.\n");
		wait_for_key_then_exit();
	}
	
	for (size_t j = 0; j < l->n_anchors; j++) {
		float w = l->anchors[j].w;  // percentage of image width
		float h = l->anchors[j].h;  // percentage of image height
		l->anchors[j].area = w * h;
		l->anchors[j].cx = 0.0F;  // percentage of cell size
		l->anchors[j].cy = 0.0F;
		l->anchors[j].left = -w / 2.0F;
		l->anchors[j].right = l->anchors[j].left + w;
		l->anchors[j].top = -h / 2.0F;
		l->anchors[j].bottom = l->anchors[j].top + h;
	}
	for (size_t k = 0; k < net->n_anchors; k++) {
		if (!net->anchors[k].w) {
			for (size_t j = 0; j < l->n_anchors; j++) {
				if (k + j >= net->n_anchors) {
					printf("Out of bounds error when settings network anchors.\n");
					wait_for_key_then_exit();
				}
				net->anchors[k + j] = l->anchors[j];
				net->anchors[k + j].lbl = l->id;
			}
			break;
		}
	}
	l->detections = (bbox*)xcalloc(l->in.w * l->in.h * l->n_anchors, sizeof(bbox));
	l->sorted = (bbox**)xcalloc(l->in.w * l->in.h * l->n_anchors, sizeof(bbox*));
	// TODO: Make these cfg parameters
	l->cull_thresh = 0.5F;
	l->nms_iou_thresh = 0.5F;
	l->ignore_thresh = 0.7F;
	l->iou_thresh = 0.2F;
	l->obj_normalizer = 0.4F;
	l->max_box_grad = 1.0F;
	l->scale_grid = 2.0F;
}

void set_activate(layer* l, int use_gpu) {
	if (use_gpu) {
#ifndef GPU
		gpu_not_defined();
#else
		switch (l->activation) {
		case ACT_RELU:
			l->activate = activate_relu_gpu;
			break;
		case ACT_MISH:
			l->activate = activate_mish_gpu;
			break;
		case ACT_SIGMOID:
			l->activate = activate_sigmoid_gpu;
			break;
		case ACT_LEAKY:
			l->activate = activate_leaky_relu_gpu;
			break;
		case ACT_TANH:
			l->activate = activate_tanh_gpu;
			break;
		case ACT_SOFTMAX:
			l->activate = activate_softmax_gpu;
			break;
		default:
			l->activate = activate_none;
		}
#endif
	}
	else {
		switch (l->activation) {
		case ACT_RELU:
			l->activate = activate_relu;
			break;
		case ACT_MISH:
			l->activate = activate_mish;
			break;
		case ACT_SIGMOID:
			l->activate = activate_sigmoid;
			break;
		case ACT_LEAKY:
			l->activate = activate_leaky_relu;
			break;
		case ACT_TANH:
			l->activate = activate_tanh;
			break;
		case ACT_SOFTMAX:
			l->activate = activate_softmax;
			break;
		default:
			l->activate = activate_none;
		}
	}
}

void activate(float* Z, float* output, size_t out_n, size_t batch_size, ACTIVATION act, int use_gpu) {
	if (use_gpu) {
#ifndef GPU
		gpu_not_defined();
#else
		switch (act) {
		case ACT_RELU:
			activate_relu_gpu(Z, output, out_n, batch_size);
			break;
		case ACT_MISH:
			activate_mish_gpu(Z, output, out_n, batch_size);
			break;
		case ACT_SIGMOID:
			activate_sigmoid_gpu(Z, output, out_n, batch_size);
			break;
		case ACT_LEAKY:
			activate_leaky_relu_gpu(Z, output, out_n, batch_size);
			break;
		case ACT_TANH:
			activate_tanh_gpu(Z, output, out_n, batch_size);
			break;
		case ACT_SOFTMAX:
			activate_softmax_gpu(Z, output, out_n, batch_size);
			break;
		default:
			activate_none(Z, output, out_n, batch_size);
		}
#endif
	}
	else {
		switch (act) {
		case ACT_RELU:
			activate_relu(Z, output, out_n, batch_size);
			break;
		case ACT_MISH:
			activate_mish(Z, output, out_n, batch_size);
			break;
		case ACT_SIGMOID:
			activate_sigmoid(Z, output, out_n, batch_size);
			break;
		case ACT_LEAKY:
			activate_leaky_relu(Z, output, out_n, batch_size);
			break;
		case ACT_TANH:
			activate_tanh(Z, output, out_n, batch_size);
			break;
		case ACT_SOFTMAX:
			activate_softmax(Z, output, out_n, batch_size);
			break;
		default:
			activate_none(Z, output, out_n, batch_size);
		}
	}
}

void get_activation_grads(layer* l, size_t batch_size) {
	if (l->activation == ACT_MISH) get_grads_mish(l->grads, l->act_inputs, l->out.n, batch_size);
	else if (l->activation == ACT_RELU) get_grads_relu(l->grads, l->act_inputs, l->out.n, batch_size);
	else if (l->activation == ACT_LEAKY) get_grads_leaky_relu(l->grads, l->act_inputs, l->out.n, batch_size);
	else if (l->activation == ACT_SIGMOID) get_grads_sigmoid(l->grads, l->output, l->out.n, batch_size);
	else if (l->activation == ACT_SOFTMAX);
	else if (l->activation == ACT_TANH) get_grads_tanh(l->grads, l->act_inputs, l->out.n, batch_size);
	else if (l->activation == ACT_NONE);
	else {
		printf("Invalid activation function.");
		wait_for_key_then_exit();
	}
}

#pragma warning (suppress:4100)
void get_activation_grads_gpu(layer* l, size_t batch_size) {
#ifdef GPU
	if (l->activation == ACT_MISH) get_grads_mish_gpu(l->gpu.grads, l->gpu.act_inputs, (int)l->out.n, (int)batch_size);
	else if (l->activation == ACT_RELU) get_grads_relu_gpu(l->gpu.grads, l->gpu.act_inputs, (int)l->out.n, (int)batch_size);
	else if (l->activation == ACT_LEAKY) get_grads_leaky_relu_gpu(l->gpu.grads, l->gpu.act_inputs, (int)l->out.n, (int)batch_size);
	else if (l->activation == ACT_SIGMOID) get_grads_sigmoid_gpu(l->gpu.grads, l->gpu.output, (int)l->out.n, (int)batch_size);
	else if (l->activation == ACT_SOFTMAX);
	else if (l->activation == ACT_TANH) get_grads_tanh_gpu(l->gpu.grads, l->gpu.act_inputs, (int)l->out.n, (int)batch_size);
	else if (l->activation == ACT_NONE);
	else {
		printf("Invalid activation function.");
		wait_for_key_then_exit();
	}
#else
	gpu_not_defined();
#endif
}

void set_loss(layer* l, int use_gpu) {
	if (use_gpu) {
#ifndef GPU
		gpu_not_defined();
#else
		switch (l->loss_type) {
		case LOSS_MAE:
			l->get_loss = loss_mae_gpu;
			break;
		case LOSS_MSE:
			l->get_loss = loss_mse_gpu;
			break;
		case LOSS_BCE:
			l->get_loss = loss_bce_gpu;
			break;
		case LOSS_CCE:
			l->get_loss = loss_cce_gpu;
			break;
		default:
			printf("No loss function set. Layer %d\n", l->id);
		}
#endif
	}
	else {
		switch (l->loss_type) {
		case LOSS_MAE:
			l->get_loss = loss_mae;
			break;
		case LOSS_MSE:
			l->get_loss = loss_mse;
			break;
		case LOSS_BCE:
			l->get_loss = loss_bce;
			break;
		case LOSS_CCE:
			l->get_loss = loss_cce;
			break;
		default:
			printf("No loss function set. Layer %d\n", l->id);
		}
	}
}

void set_regularization(network* net) {
	if (net->use_gpu) {
		CUDA_MALLOC(&net->gpu.reg_loss, 1, sizeof(float));
	}
	if (net->regularization == REG_L1) {
		if (net->use_gpu) {
			net->get_reg_loss = reg_loss_l1_gpu;
			net->regularize_weights = regularize_l1_gpu;
		}
		else {
			net->get_reg_loss = reg_loss_l1;
			net->regularize_weights = regularize_l1;
		}
	}
	else if (net->regularization == REG_L2) {
		if (net->use_gpu) {
			net->get_reg_loss = reg_loss_l2_gpu;
			net->regularize_weights = regularize_l2_gpu;
		}
		else {
			net->get_reg_loss = reg_loss_l2;
			net->regularize_weights = regularize_l2;
		}
	}
	else {
		net->regularize_weights = regularize_none;
	}
}

#pragma warning(suppress:4100)  // unreferenced formal parameter: 'net'
void backward_none(layer* l, network* net) {
}

#pragma warning(suppress:4100)  // unreferenced formal parameter: 'net'
void update_none(layer* l, network* net) {
}

void free_network(network* n) {
	for (size_t i = 0; i < n->n_classes; i++) {
		xfree(&n->class_names[i]);
	}
	xfree(&n->class_names);
	xfree(&n->step_percents.a);
	xfree(&n->step_scaling.a);
	free_layer_fields(n->input, n);
	xfree(&n->input);
	free_network_layers(n);
	xfree(&n->workspace);
	xfree(&n->dataset_dir);
	xfree(&n->weights_file);
	xfree(&n->save_dir);
	if (n->type == NET_CLASSIFY) {
		free_classifier_dataset(n->data.classifier);
	}
	else if (n->type == NET_DETECT) {
		free_detector_dataset(n->data.detector);
		xfree(&n->anchors);
	}
	if (n->use_gpu) {
		CUDA_FREE(&n->gpu.workspace);
		CUDA_FREE(&n->gpu.reg_loss);
	}
	xfree(&n);
}

void free_network_layers(network* net) {
	for (size_t i = 0; i < net->n_layers; i++) {
		free_layer_fields(&net->layers[i], net);
	}
	xfree(&net->layers);
}

void free_layer_fields(layer* l, network* net) {
	xfree(&l->inputs);
	xfree(&l->anchors);
	xfree(&l->biases);
	xfree(&l->bias_grads);
	xfree(&l->bias_velocities);
	xfree(&l->detections);
	xfree(&l->errors);
	xfree(&l->grads);
	xfree(&l->maxpool_addresses);
	xfree(&l->output);
	xfree(&l->sorted);
	xfree(&l->truth);
	xfree(&l->weights);
	xfree(&l->weight_grads);
	xfree(&l->weight_velocities);
	xfree(&l->Z);
	if (l->batchnorm) {
		xfree(&l->act_inputs);
		xfree(&l->gammas);
		xfree(&l->gamma_grads);
		xfree(&l->gamma_velocities);
		xfree(&l->means);
		xfree(&l->rolling_means);
		xfree(&l->rolling_variances);
		xfree(&l->variances);
		xfree(&l->Z_norm);
	}
	if (net->use_gpu) {
		CUDA_FREE(&l->gpu.anchors);
		CUDA_FREE(&l->gpu.biases);
		CUDA_FREE(&l->gpu.bias_grads);
		CUDA_FREE(&l->gpu.bias_velocities);
		CUDA_FREE(&l->gpu.detections);
		CUDA_FREE(&l->gpu.errors);
		CUDA_FREE(&l->gpu.grads);
		CUDA_FREE(&l->gpu.loss);
		CUDA_FREE(&l->gpu.maxpool_addresses);
		CUDA_FREE(&l->gpu.output);
		CUDA_FREE(&l->gpu.sorted);
		CUDA_FREE(&l->gpu.truth);
		CUDA_FREE(&l->gpu.weights);
		CUDA_FREE(&l->gpu.weight_grads);
		CUDA_FREE(&l->gpu.weight_velocities);
		CUDA_FREE(&l->gpu.Z);
		if (l->batchnorm) {
			CUDA_FREE(&l->gpu.act_inputs);
			CUDA_FREE(&l->gpu.gammas);
			CUDA_FREE(&l->gpu.gamma_grads);
			CUDA_FREE(&l->gpu.gamma_velocities);
			CUDA_FREE(&l->gpu.means);
			CUDA_FREE(&l->gpu.rolling_means);
			CUDA_FREE(&l->gpu.rolling_variances);
			CUDA_FREE(&l->gpu.variances);
			CUDA_FREE(&l->gpu.Z_norm);
		}
	}
}

size_t get_train_params_count(network* net) {
	size_t n_layers = net->n_layers;
	layer* ls = net->layers;
	size_t sum = 0;
	size_t i;
#pragma omp parallel for reduction(+:sum)
	for (i = 0; i < n_layers; i++) {
		sum += get_layer_param_count(&ls[i]);
	}
	return sum;
}

size_t get_layer_param_count(layer* l) {
	size_t N = l->n_weights;
	N += l->k.n_filters;  // # of biases
	if (l->batchnorm) N += l->out.c * 3;  // gammas, rolling means & variances
	return N;
}


/*** PRINTS ***/


void print_network(network* n) {
	printf("\n[NETWORK]\n\n");
	printf("n_layers: %zu\n", n->n_layers);
	printf("w, h, c: %zu, %zu, %zu\n", n->w, n->h, n->c);
	printf("batch_size: %zu\n", n->batch_size);
	printf("subbatch_size: %zu\n", n->subbatch_size);
	printf("max_iterations: %zu\n", n->max_iterations);
	printf("learning_rate: %f\n", n->learning_rate);
	printf("lr_policy: ");
	print_lrpolicy(n->lr_policy);
	printf("step_percents: ");
	print_floatarr(&(n->step_percents));
	printf("step_scaling: ");
	print_floatarr(&(n->step_scaling));
	printf("ease_in: %zu\n", n->ease_in);
	printf("momentum: %f\n", n->momentum);
	printf("decay: %f\n", n->decay);
	printf("saturation: %f, %f\n", n->saturation[0], n->saturation[1]);
	printf("exposure: %f, %f\n", n->exposure[0], n->exposure[1]);
	printf("hue: %f, %f\n", n->hue[0], n->hue[1]);
	print_layers(n);
	printf("[END NETWORK]\n");
}

void print_layers(network* net) {
	size_t n = net->n_layers;
	layer* ls = net->layers;
	for (size_t i = 0; i < n; i++) {
		print_layer(&ls[i]);
	}
}

void print_layer(layer* l) {
	if (l->type == LAYER_CONV) print_layer_conv(l);
	else if (l->type == LAYER_CLASSIFY) print_layer_classify(l);
	else if (l->type == LAYER_MAXPOOL) print_layer_maxpool(l);
	else if (l->type == LAYER_DENSE) print_layer_conv(l);
	else if (l->type == LAYER_RESIDUAL) print_layer_residual(l);
	else printf("Not Implemented\n");
}

void print_layer_conv(layer* l) {
	printf("\n[LAYER]\n");
	printf("id: %d\n", l->id);
	printf("layer_type: ");
	print_layertype(l->type);
	printf("activation: ");
	print_activation(l->activation);
	printf("batchnorm: %d\n", l->batchnorm);
	printf("w, h, c: %zu, %zu, %zu\n", l->in.w, l->in.h, l->in.c);
	printf("n_filters: %zu\n", l->k.n_filters);
	printf("ksize: %zu\n", l->k.ksize);
	printf("stride: %zu\n", l->k.stride);
	printf("pad: %zu\n", l->k.pad);
	printf("# of inputs: %zu\n", l->in.n);
	printf("# of outputs: %zu\n", l->out.n);
	printf("# of weights: %zu\n", l->n_weights);
	printf("train: %d\n", l->train);
	printf("input ids: ");
	for (int i = 0; i < l->n_inputs; i++) {
		printf("%d", l->inputs[i].source_id);
		if (i < l->n_inputs - 1) printf(", ");
		else printf("\n");
	}
	printf("[END LAYER]\n");
}

void print_layer_classify(layer* l) {
	printf("\n[LAYER]\n");
	printf("id: %d\n", l->id);
	printf("layer_type: ");
	print_layertype(l->type);
	printf("activation: ");
	print_activation(l->activation);
	printf("loss: ");
	print_loss_type(l->loss_type);
	printf("# of classes: %zu\n", l->n_classes);
	printf("w, h, c: %zu, %zu, %zu\n", l->in.w, l->in.h, l->in.c);
	printf("stride: %zu\n", l->k.stride);
	printf("# of inputs: %zu\n", l->in.n);
	printf("# of outputs: %zu\n", l->out.n);
	printf("# of weights: %zu\n", l->n_weights);
	printf("train: %d\n", l->train);
	printf("input ids: ");
	for (int i = 0; i < l->n_inputs; i++) {
		printf("%d", l->inputs[i].source_id);
		if (i < l->n_inputs - 1) printf(", ");
		else printf("\n");
	}
	printf("[END LAYER]\n");
}

void print_layer_maxpool(layer* l) {
	printf("\n[LAYER]\n");
	printf("id: %d\n", l->id);
	printf("layer_type: ");
	print_layertype(l->type);
	printf("w, h, c: %zu, %zu, %zu\n", l->in.w, l->in.h, l->in.c);
	printf("ksize: %zu\n", l->k.ksize);
	printf("stride: %zu\n", l->k.stride);
	printf("pad: %zu\n", l->k.pad);
	printf("# of inputs: %zu\n", l->in.n);
	printf("# of outputs: %zu\n", l->out.n);
	printf("train: %d\n", l->train);
	printf("input ids: ");
	for (int i = 0; i < l->n_inputs; i++) {
		printf("%d", l->inputs[i].source_id);
		if (i < l->n_inputs - 1) printf(", ");
		else printf("\n");
	}
	printf("[END LAYER]\n");
}

void print_layer_residual(layer* l) {
	printf("\n[LAYER]\n");
	printf("id: %d\n", l->id);
	printf("layer_type: ");
	print_layertype(l->type);
	printf("activation: ");
	print_activation(l->activation);
	printf("w, h, c: %zu, %zu, %zu\n", l->in.w, l->in.h, l->in.c);
	printf("# of inputs: %zu\n", l->in.n);
	printf("# of outputs: %zu\n", l->out.n);
	printf("input ids: ");
	for (int i = 0; i < l->n_inputs; i++) {
		printf("%d", l->inputs[i].source_id);
		if (i < l->n_inputs - 1) printf(", ");
		else printf("\n");
	}
	printf("[END LAYER]\n");
}

void print_lrpolicy(LR_POLICY lrp) {
	if (lrp == LR_STEPS) printf("steps\n");
	else printf("NONE\n");
}

void print_layertype(LAYER_TYPE lt) {
	if (lt == LAYER_NONE) printf("none\n");
	else if (lt == LAYER_AVGPOOL_GLOBAL) printf("avgpool global\n");
	else if (lt == LAYER_AVGPOOL_LOCAL) printf("avgpool local\n");
	else if (lt == LAYER_CLASSIFY) printf("classify\n");
	else if (lt == LAYER_CONV) printf("conv\n");
	else if (lt == LAYER_DETECT) printf("detect\n");
	else if (lt == LAYER_DENSE) printf("dense\n");
	else if (lt == LAYER_MAXPOOL) printf("maxpool\n");
	else if (lt == LAYER_RESIDUAL) printf("residual\n");
	else if (lt == LAYER_UPSAMPLE) printf("upsample\n");
	else printf("INVALID\n");
}

void print_activation(ACTIVATION a) {
	if (a == ACT_NONE) printf("none\n");
	else if (a == ACT_RELU) printf("relu\n");
	else if (a == ACT_LEAKY) printf("leaky relu\n");
	else if (a == ACT_MISH) printf("mish\n");
	else if (a == ACT_SIGMOID) printf("sigmoid\n");
	else if (a == ACT_SOFTMAX) printf("softmax\n");
	else if (a == ACT_TANH) printf("tanh\n");
	else printf("INVALID\n");
}

void print_loss_type(LOSS_TYPE c) {
	if (c == LOSS_NONE) printf("none\n");
	else if (c == LOSS_MAE) printf("mae\n");
	else if (c == LOSS_MSE) printf("mse\n");
	else if (c == LOSS_CCE) printf("cce\n");
	else printf("INVALID\n");
}

void print_all_network_weights(network* net) {
	layer* layers = net->layers;
	size_t N = net->n_layers;
	size_t n;
	layer* l;
	printf("\nWEIGHTS");
	for (size_t i = 0; i < N; i++) {
		printf("\n\nLAYER %zu\n", i);
		l = &layers[i];
		n = l->n_weights;
		for (size_t j = 0; j < n; j++) {
			printf("%f, ", l->weights[j]);
			if ((j + 1) % 10 == 0) printf("\n");
		}
	}
}

void print_some_weights(layer* l, size_t n) {
	float* weights = l->weights;
	assert(l->n_weights >= n);
	for (size_t i = 0; i < n; i++) {
		printf("%f\n", weights[i]);
	}
}

void print_top_class_name(float* probs, size_t n_classes, char** class_names, int include_prob, int new_line) {
	size_t c = 0;
	float highscore = 0.0F;
	for (size_t i = 0; i < n_classes; i++) {
		float p = probs[i];
		if (p > highscore) {
			highscore = p;
			c = i;
		}
	}
	char end = new_line ? '\n' : '\0';
	if (include_prob) printf("%s (%f)%c", class_names[c], highscore, end);
	else printf("%s%c", class_names[c], end);
}

void print_network_summary(network* net, int print_training_params) {
	printf("[NETWORK SUMMARY]\n"
		"Cfg: %s\n"
		"Classes: %zu\n"
		"Input: %zux%zux%zu\n"
		"Layers: %zu\n",
		net->cfg_file, net->n_classes, net->w, net->h, net->c, net->n_layers);
	if (print_training_params) {
		printf("Batch size: %zu\n"
			"Learning rate: %f\n"
			"Momentum: %f\n"
			"Iterations: %zu\n"
			"# of trainable params: %zu\n",
			net->batch_size, net->learning_rate, net->momentum, net->max_iterations, get_train_params_count(net));
	}
	printf("\n");
}

void print_network_structure(network* net) {
	size_t n_layers = net->n_layers;
	layer* layers = net->layers;
	char ltbuf[50] = { 0 };
	char idbuf[50] = { 0 };
	for (size_t i = 0; i < n_layers; i++) {
		layer* l = &layers[i];
		get_layer_type_str(ltbuf, sizeof(ltbuf), l->type);
		get_input_ids_str(idbuf, 50, l->inputs, l->n_inputs);
		printf("[%d] %s inputs[%s] %zux%zux%zu -> %zux%zux%zu pad: %zu stride: %zu ksize: %zu\n", l->id, ltbuf, idbuf, l->in.w, l->in.h, l->in.c, l->out.w, l->out.h, l->out.c, l->k.pad, l->k.stride, l->k.ksize);
	}
}

void print_prediction_results(network* net, layer* prediction_layer) {
	size_t n_classes = net->n_classes;
	char** class_names = net->class_names;
	float* predictions = prediction_layer->output;
	float* truth = prediction_layer->truth;
	printf("Truth:     ");
	print_top_class_name(truth, n_classes, class_names, 1, 1);
	printf("Predicted: ");
	print_top_class_name(predictions, n_classes, class_names, 1, 1);
	printf("Class loss = %f\n", prediction_layer->loss);
}

void get_layer_type_str(char* buf, size_t bufsize, LAYER_TYPE lt) {
	if (bufsize < 50) {
		printf("Error: bufsize too small.\n");
		wait_for_key_then_exit();
	}
	memset(buf, 0, bufsize);
	if (lt == LAYER_NONE) strcpy(buf, "none");
	else if (lt == LAYER_AVGPOOL_GLOBAL) strcpy(buf, "avgpool");
	else if (lt == LAYER_CLASSIFY) strcpy(buf, "classify");
	else if (lt == LAYER_CONV) strcpy(buf, "conv");
	else if (lt == LAYER_DETECT) strcpy(buf, "detect");
	else if (lt == LAYER_DENSE) strcpy(buf, "dense");
	else if (lt == LAYER_MAXPOOL) strcpy(buf, "maxpool");
	else if (lt == LAYER_RESIDUAL) strcpy(buf, "residual");
	else if (lt == LAYER_UPSAMPLE) strcpy(buf, "upsample");
}

void get_input_ids_str(char* buf, size_t bufsize, layer_input* inputs, size_t n_inputs) {
	if (bufsize < 50) {
		printf("Error: bufsize too small.\n");
		wait_for_key_then_exit();
	}
	memset(buf, 0, bufsize);
	if (!inputs) return;
	snprintf(buf, bufsize, "%d", inputs[0].source_id);
	for (size_t i = 1; i < n_inputs; i++) {
		size_t length = strlen(buf);
		snprintf(&buf[length], bufsize - length, ",%d", inputs[i].source_id);
	}
}