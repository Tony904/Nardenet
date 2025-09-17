#include "network.h"
#include <assert.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include "utils.h"
#include "xallocs.h"
#include "layer_conv.h"
#include "layer_classify.h"
#include "layer_fc.h"
#include "layer_maxpool.h"
#include "layer_residual.h"
#include "layer_detect.h"
#include "layer_avgpool.h"
#include "layer_upsample.h"
#include "layer_route.h"
#include "activations.h"
#include "loss.h"
#include "derivatives.h"
#include "data.h"
#include "blas.h"
#include "xcuda.h"

#pragma warning (disable:4702)

void build_input_layer(network* net);
void build_layer(int i, network* net);
void build_conv_layer(int i, network* net);
void build_fc_layer(int i, network* net);
void build_maxpool_layer(int i, network* net);
void build_avgpool_layer(int i, network* net);
void build_residual_layer(int i, network* net);
void build_upsample_layer(int i, network* net);
void build_classify_layer(int i, network* net);
void build_detect_layer(int i, network* net);

void set_activate(layer* l, int use_gpu);
void set_loss(layer* l, int use_gpu);
void set_regularization(network* net);

void backward_none(layer* l, network* net);
void update_none(layer* l, network* net);

size_t get_train_params_count(network* net);
size_t get_layer_param_count(layer* l);

void free_network(network* net);
void free_network_layers(network* net);
void free_layer_members(layer* l);

void print_layer_conv(layer* l);
void print_layer_classify(layer* l);
void print_layer_maxpool(layer* l);
void print_layer_residual(layer* l);

void get_layer_type_str(char* buf, size_t bufsize, LAYER_TYPE lt);
void get_in_ids_str(char* buf, size_t bufsize, intarr inputs);


network* new_network(size_t num_of_layers) {
	network* net = (network*)xcalloc(1, sizeof(network));
	net->input = (layer*)xcalloc(1, sizeof(layer));
	net->n_layers = num_of_layers;
	net->layers = (layer*)xcalloc(num_of_layers, sizeof(layer));
	return net;
}

void build_network(network* net) {
	if (net->type == NET_DETECT) {
		net->data.detr.current_batch = (det_sample**)xcalloc(net->batch_size, sizeof(det_sample*));
		net->anchors = (bbox*)xcalloc(net->n_anchors, sizeof(bbox));
	}
	build_input_layer(net);
	size_t largest_workspace = 0;
	layer* ls = net->layers;
	for (int i = 0; i < net->n_layers; i++) {
		build_layer(i, net);
		layer* l = &ls[i];
		size_t ws_size = l->out_w * l->out_h * l->ksize * l->ksize * l->c;
		largest_workspace = max(largest_workspace, ws_size);
	}
	net->workspace = (float*)xcalloc(largest_workspace, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&net->gpu.workspace, largest_workspace * sizeof(float));
	}
	net->workspace_size = largest_workspace;
	net->current_learning_rate = net->learning_rate;
	set_regularization(net);
}

void build_layer(int i, network* net) {
	layer* l = &(net->layers[i]);
	assert(l->type != LAYER_NONE);
	if (l->type == LAYER_CONV) build_conv_layer(i, net);
	else if (l->type == LAYER_CLASSIFY) build_classify_layer(i, net);
	else if (l->type == LAYER_MAXPOOL) build_maxpool_layer(i, net);
	else if (l->type == LAYER_FC) build_fc_layer(i, net);
	else if (l->type == LAYER_RESIDUAL) build_residual_layer(i, net);
	else if (l->type == LAYER_DETECT) build_detect_layer(i, net);
	else if (l->type == LAYER_AVGPOOL) build_avgpool_layer(i, net);
	else if (l->type == LAYER_UPSAMPLE) build_upsample_layer(i, net);
	else {
		printf("Unknown layer type: %d\n", (int)l->type);
		wait_for_key_then_exit();
	}
}

void build_input_layer(network* net) {
	layer* l = net->input;
	l->type = LAYER_INPUT;
	l->train = 0;
	l->id = -1;
	l->w = net->w;
	l->h = net->h;
	l->c = net->c;
	l->n = l->w * l->h * l->c;
	l->out_w = l->w;
	l->out_h = l->h;
	l->out_c = l->c;
	l->out_n = l->n;
	l->output = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.output, l->out_n * net->batch_size * sizeof(float));
	}
}

/* i = layer index in net->layers */
void build_conv_layer(int i, network* net) {
	layer* l = &(net->layers[i]);
	layer* ls = net->layers;
	l->id = i;
	if (l->n_groups < 1) l->n_groups = 1;
	if (l->n_filters % l->n_groups > 0) {
		printf("Cannot evenly distribute %zu filters between %zu groups. (layer %d)\n", l->n_filters, l->n_groups, i);
		wait_for_key_then_exit();
	}

	// Set default in_ids if none specified.
	if (l->in_ids.n == 0) {
		l->in_ids.a = (int*)xcalloc(1, sizeof(int));
		l->in_ids.a[0] = i - 1;
		l->in_ids.n = 1;
	}
	else {
		for (int j = 0; j < l->in_ids.n; j++) {
			if (l->in_ids.a[j] < 0) l->in_ids.a[j] += i;
			if (l->in_ids.a[j] > i || l->in_ids.a[j] < 0) {
				printf("Invalid in_id of %d for layer %d\n", l->in_ids.a[j], i);
				wait_for_key_then_exit();
			}
		}
	}

	// Build array of input layer addresses.
	l->in_layers = (layer**)xcalloc(l->in_ids.n, sizeof(layer*));
	if (i > 0) {
		for (size_t j = 0; j < l->in_ids.n; j++) {
			l->in_layers[j] = &ls[l->in_ids.a[j]];
		}
	}
	else { // if first layer
		l->in_layers[0] = net->input;
	}
	
	// Calculate input dimensions.
	l->w = l->in_layers[0]->out_w;
	l->h = l->in_layers[0]->out_h;
	l->c = l->in_layers[0]->out_c;
	for (size_t j = 1; j < l->in_ids.n; j++) {
		layer* inl = l->in_layers[j];
		if (l->w != inl->out_w || l->h != inl->out_h) {
			printf("Invalid input layer dimensions. Width and height must match.\n");
			printf("Layer %d %zux%zu, Input layer %d %zux%zu\n", i, l->w, l->h, inl->id, inl->out_w, inl->out_h);
			wait_for_key_then_exit();
		}
		l->c += inl->out_c;
	}
	l->n = l->w * l->h * l->c;

	if (l->c % l->n_groups > 0) {
		printf("Cannot evenly distribute %zu channels between %zu groups. (layer %d)\n", l->c, l->n_groups, i);
		wait_for_key_then_exit();
	}

	if (((l->w + (l->pad * 2)) - l->ksize) % l->stride != 0) {
		printf("Invalid kernel size or stride for input width.\n");
		printf("Layer %d width %zu (w/ padding), kernel size %zu, stride %zu\n", i, l->w + l->pad, l->ksize, l->stride);
		wait_for_key_then_exit();
	}
	if (((l->h + (l->pad * 2)) - l->ksize) % l->stride != 0) {
		printf("Invalid kernel size or stride for input height.\n");
		printf("Layer %d height %zu (w/ padding), kernel size %zu, stride %zu\n", i, l->h + l->pad, l->ksize, l->stride);
		wait_for_key_then_exit();
	}

	// Calculate output dimensions.
	l->out_w = ((l->w + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_h = ((l->h + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_c = l->n_filters;
	l->out_n = l->out_w * l->out_h * l->out_c;

	l->Z = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	l->n_weights = l->n_filters * l->ksize * l->ksize * (l->c / l->n_groups);
	l->weights = (float*)xcalloc(l->n_weights, sizeof(float));
	l->biases = (float*)xcalloc(l->n_filters, sizeof(float));
	l->grads = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	l->weight_grads = (float*)xcalloc(l->n_weights, sizeof(float));
	l->bias_grads = (float*)xcalloc(l->n_filters, sizeof(float));
	l->weight_velocities = (float*)xcalloc(l->n_weights, sizeof(float));
	l->bias_velocities = (float*)xcalloc(l->n_filters, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.Z, l->out_n * net->batch_size * sizeof(float));
		CUDA_MALLOC(&l->gpu.weights, l->n_weights * sizeof(float));
		CUDA_MALLOC(&l->gpu.biases, l->n_filters * sizeof(float));
		CUDA_MALLOC(&l->gpu.grads, l->out_n * net->batch_size * sizeof(float));
		CUDA_MALLOC(&l->gpu.weight_grads, l->n_weights * sizeof(float));
		CUDA_MALLOC(&l->gpu.bias_grads, l->n_filters * sizeof(float));
		CUDA_MALLOC(&l->gpu.weight_velocities, l->n_weights * sizeof(float));
		CUDA_MALLOC(&l->gpu.bias_velocities, l->n_filters * sizeof(float));
	}
	if (l->batchnorm) {
		l->Z_norm = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
		l->act_inputs = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
		l->means = (float*)xcalloc(l->n_filters, sizeof(float));
		l->variances = (float*)xcalloc(l->n_filters, sizeof(float));
		l->gammas = (float*)xcalloc(l->n_filters, sizeof(float));
		fill_array(l->gammas, l->n_filters, 1.0F);
		l->gamma_grads = (float*)xcalloc(l->n_filters, sizeof(float));
		l->gamma_velocities = (float*)xcalloc(l->n_filters, sizeof(float));
		l->rolling_means = (float*)xcalloc(l->n_filters, sizeof(float));
		l->rolling_variances = (float*)xcalloc(l->n_filters, sizeof(float));
		if (net->use_gpu) {
			CUDA_MALLOC(&l->gpu.Z_norm, l->out_n * net->batch_size * sizeof(float));
			CUDA_MALLOC(&l->gpu.act_inputs, l->out_n * net->batch_size * sizeof(float));
			CUDA_MALLOC(&l->gpu.means, l->n_filters * sizeof(float));
			CUDA_MALLOC(&l->gpu.variances, l->n_filters * sizeof(float));
			CUDA_MALLOC(&l->gpu.gammas, l->n_filters * sizeof(float));
			CUDA_MEMCPY_H2D(l->gpu.gammas, l->gammas, l->n_filters * sizeof(float));
			CUDA_MALLOC(&l->gpu.gamma_grads, l->n_filters * sizeof(float));
			CUDA_MALLOC(&l->gpu.gamma_velocities, l->n_filters * sizeof(float));
			CUDA_MALLOC(&l->gpu.rolling_means, l->n_filters * sizeof(float));
			CUDA_MALLOC(&l->gpu.rolling_variances, l->n_filters * sizeof(float));
		}
	}
	else {
		l->act_inputs = l->Z;
		l->gpu.act_inputs = l->gpu.Z;
	}
	if (l->activation) {
		l->output = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
		if (net->use_gpu) {
			CUDA_MALLOC(&l->gpu.output, l->out_n * net->batch_size * sizeof(float));
		}
	}
	else {
		l->output = l->act_inputs;
		l->gpu.output = l->gpu.act_inputs;
	}
	
	if (net->use_gpu) {
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

void build_fc_layer(int i, network* net) {
	layer* l = &(net->layers[i]);
	layer* ls = net->layers;
	l->id = i;

	if (l->in_ids.n == 0) {
		l->in_ids.a = (int*)xcalloc(1, sizeof(int));
		l->in_ids.a[0] = i - 1;
		l->in_ids.n = 1;
	}
	else {
		for (int j = 0; j < l->in_ids.n; j++) {
			if (l->in_ids.a[j] < 0) l->in_ids.a[j] += i;
			if (l->in_ids.a[j] > i || l->in_ids.a[j] < 0) {
				printf("Invalid in_id of %d for layer %d\n", l->in_ids.a[j], i);
				wait_for_key_then_exit();
			}
		}
	}

	l->in_layers = (layer**)xcalloc(l->in_ids.n, sizeof(layer*));
	for (size_t j = 0; j < l->in_ids.n; j++) {
		l->in_layers[j] = &ls[l->in_ids.a[j]];
	}

	l->w = l->in_layers[0]->out_w;
	l->h = l->in_layers[0]->out_h;
	l->c = l->in_layers[0]->out_c;
	for (size_t j = 1; j < l->in_ids.n; j++) {
		layer* inl = l->in_layers[i];
		if (l->w != inl->out_w || l->h != inl->out_h) {
			printf("Invalid input layer dimensions. Width and height must match.\n");
			printf("Layer %d %zux%zu, Input layer %d %zux%zu\n", i, l->w, l->h, inl->id, inl->out_w, inl->out_h);
			wait_for_key_then_exit();
		}
		l->c += inl->out_c;
	}
	l->n = l->w * l->h * l->c;

	l->pad = 0;
	l->stride = 1;
	assert(l->w == l->h);
	l->ksize = l->w;

	// Calculate output dimensions.
	l->out_w = ((l->w + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_h = ((l->h + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_c = l->n_filters;
	l->out_n = l->out_w * l->out_h * l->out_c;

	l->Z = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	l->n_weights = l->n_filters * l->ksize * l->ksize * l->c;
	l->weights = (float*)xcalloc(l->n_weights, sizeof(float));
	l->biases = (float*)xcalloc(l->n_filters, sizeof(float));
	l->grads = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	l->weight_grads = (float*)xcalloc(l->n_weights, sizeof(float));
	l->bias_grads = (float*)xcalloc(l->n_filters, sizeof(float));
	l->weight_velocities = (float*)xcalloc(l->n_weights, sizeof(float));
	l->bias_velocities = (float*)xcalloc(l->n_filters, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.Z, l->out_n * net->batch_size * sizeof(float));
		CUDA_MALLOC(&l->gpu.weights, l->n_weights * sizeof(float));
		CUDA_MALLOC(&l->gpu.biases, l->n_filters * sizeof(float));
		CUDA_MALLOC(&l->gpu.grads, l->out_n * net->batch_size * sizeof(float));
		CUDA_MALLOC(&l->gpu.weight_grads, l->n_weights * sizeof(float));
		CUDA_MALLOC(&l->gpu.bias_grads, l->n_filters * sizeof(float));
		CUDA_MALLOC(&l->gpu.weight_velocities, l->n_weights * sizeof(float));
		CUDA_MALLOC(&l->gpu.bias_velocities, l->n_filters * sizeof(float));
	}
	if (l->batchnorm) {
		l->Z_norm = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
		l->act_inputs = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
		l->means = (float*)xcalloc(l->n_filters, sizeof(float));
		l->variances = (float*)xcalloc(l->n_filters, sizeof(float));
		l->gammas = (float*)xcalloc(l->n_filters, sizeof(float));
		fill_array(l->gammas, l->n_filters, 1.0F);
		l->gamma_grads = (float*)xcalloc(l->n_filters, sizeof(float));
		l->gamma_velocities = (float*)xcalloc(l->n_filters, sizeof(float));
		l->rolling_means = (float*)xcalloc(l->n_filters, sizeof(float));
		l->rolling_variances = (float*)xcalloc(l->n_filters, sizeof(float));
		if (net->use_gpu) {
			CUDA_MALLOC(&l->gpu.Z_norm, l->out_n * net->batch_size * sizeof(float));
			CUDA_MALLOC(&l->gpu.act_inputs, l->out_n * net->batch_size * sizeof(float));
			CUDA_MALLOC(&l->gpu.means, l->n_filters * sizeof(float));
			CUDA_MALLOC(&l->gpu.variances, l->n_filters * sizeof(float));
			CUDA_MALLOC(&l->gpu.gammas, l->n_filters * sizeof(float));
			CUDA_MEMCPY_H2D(l->gpu.gammas, l->gammas, l->n_filters * sizeof(float));
			CUDA_MALLOC(&l->gpu.gamma_grads, l->n_filters * sizeof(float));
			CUDA_MALLOC(&l->gpu.gamma_velocities, l->n_filters * sizeof(float));
			CUDA_MALLOC(&l->gpu.rolling_means, l->n_filters * sizeof(float));
			CUDA_MALLOC(&l->gpu.rolling_variances, l->n_filters * sizeof(float));
		}
	}
	else {
		l->act_inputs = l->Z;
		l->gpu.act_inputs = l->gpu.Z;
	}

	if (l->activation) {
		l->output = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
		if (net->use_gpu) {
			CUDA_MALLOC(&l->gpu.output, l->out_n * net->batch_size * sizeof(float));
		}
	}
	else {
		l->output = l->act_inputs;
		l->gpu.output = l->gpu.act_inputs;
	}

	if (net->use_gpu) {
		l->forward = forward_fc_gpu;
		l->backward = backward_fc_gpu;
		l->update = update_conv_gpu;
	}
	else {
		l->forward = forward_fc;
		l->backward = backward_fc;
		l->update = update_conv;
	}

	set_activate(l, net->use_gpu);
}

// i = layer index in net->layers
void build_classify_layer(int i, network* net) {
	if (net->type == NET_NONE || net->type == NET_CLASSIFY) net->type = NET_CLASSIFY;
	else {
		printf("Invalid network cfg: Cannot have a classify layer in a non-classifier network.\n");
		wait_for_key_then_exit();
	}

	layer* l = &(net->layers[i]);
	layer* ls = net->layers;
	l->id = i;

	if (l->n_classes == 0) l->n_classes = net->n_classes;

	if (l->in_ids.n == 0) {
		l->in_ids.a = (int*)xcalloc(1, sizeof(int));
		l->in_ids.a[0] = i - 1;
		l->in_ids.n = 1;
	}
	else if (l->in_ids.n == 1) {
		if (l->in_ids.a[0] < 0) l->in_ids.a[0] += i;
		if (l->in_ids.a[0] > i || l->in_ids.a[0] < 0) {
			printf("Invalid in_id of %d for classify layer id %d\n", l->in_ids.a[0], i);
			wait_for_key_then_exit();
		}
	}
	else {
		printf("Classify layers can only have one input layer.\n");
		wait_for_key_then_exit();
	}

	l->in_layers = (layer**)xcalloc(l->in_ids.n, sizeof(layer*));
	for (size_t j = 0; j < l->in_ids.n; j++) {
		l->in_layers[j] = &ls[l->in_ids.a[j]];
	}

	l->w = l->in_layers[0]->out_w;
	l->h = l->in_layers[0]->out_h;
	l->c = l->in_layers[0]->out_c;
	l->n = l->w * l->h * l->c;

	l->pad = 0;
	l->stride = 1;
	if (l->w != 1 || l->h != 1 || l->c != l->n_classes) {
		printf("The layer outputting to a classify layer must have a width, height, and depth of 1x1x%zu. (is %zux%zux%zu)\n", l->n_classes, l->w, l->h, l->c);
		wait_for_key_then_exit();
	}

	// Calculate output dimensions.
	l->out_w = l->w;
	l->out_h = l->h;
	l->out_c = l->c;
	l->out_n = l->out_w * l->out_h * l->out_c;

	l->output = l->in_layers[0]->output;
	l->grads = l->in_layers[0]->grads;
	l->gpu.output = l->in_layers[0]->gpu.output;
	l->gpu.grads = l->in_layers[0]->gpu.grads;

	l->truth = (float*)xcalloc(l->n_classes * net->batch_size, sizeof(float));
	l->errors = (float*)xcalloc(l->n_classes * net->batch_size, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.truth, l->n_classes * net->batch_size * sizeof(float));
		CUDA_MALLOC(&l->gpu.errors, l->n_classes * net->batch_size * sizeof(float));
		CUDA_MALLOC(&l->gpu.loss, sizeof(float));
	}

	if (net->use_gpu) l->forward = forward_classify_gpu;
	else l->forward = forward_classify;
	l->backward = backward_none;  // all backprop stuff is done in forward pass
	l->update = update_none;

	set_loss(l, net->use_gpu);
}

/* i = layer index in net->layers */
void build_maxpool_layer(int i, network* net) {
	layer* l = &(net->layers[i]);
	layer* ls = net->layers;
	l->id = i;

	// Set default in_ids if none specified.
	if (l->in_ids.n == 0) {
		l->in_ids.a = (int*)xcalloc(1, sizeof(int));
		l->in_ids.a[0] = i - 1;
		l->in_ids.n = 1;
	}
	else {
		for (int j = 0; j < l->in_ids.n; j++) {
			if (l->in_ids.a[j] < 0) l->in_ids.a[j] += i;
			if (l->in_ids.a[j] > i || l->in_ids.a[j] < 0) {
				printf("Invalid in_id of %d for layer %d\n", l->in_ids.a[j], i);
				wait_for_key_then_exit();
			}
		}
	}

	// Build array of input layer addresses.
	l->in_layers = (layer**)xcalloc(l->in_ids.n, sizeof(layer*));
	if (i > 0) {
		for (size_t j = 0; j < l->in_ids.n; j++) {
			l->in_layers[j] = &ls[l->in_ids.a[j]];
		}
	}
	else { // if first layer
		l->in_layers[0] = net->input;
	}

	// Calculate input dimensions.
	l->w = l->in_layers[0]->out_w;
	l->h = l->in_layers[0]->out_h;
	l->c = l->in_layers[0]->out_c;
	for (size_t j = 1; j < l->in_ids.n; j++) {
		layer* inl = l->in_layers[j];
		if (l->w != inl->out_w || l->h != inl->out_h) {
			printf("Invalid input layer dimensions. Width and height must match.\n");
			printf("Layer %d %zux%zu, Input layer %d %zux%zu\n", i, l->w, l->h, inl->id, inl->out_w, inl->out_h);
			wait_for_key_then_exit();
		}
		l->c += inl->out_c;
	}
	l->n = l->w * l->h * l->c;

	// Calculate output dimensions.
	l->out_w = ((l->w - l->ksize) / l->stride) + 1;
	l->out_h = ((l->h - l->ksize) / l->stride) + 1;
	l->out_c = l->c;
	l->out_n = l->out_w * l->out_h * l->out_c;

	l->output = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	l->grads = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	l->maxpool_addresses = (float**)xcalloc(l->out_n * net->batch_size, sizeof(float*));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.output, l->out_n * net->batch_size * sizeof(float));
		CUDA_MALLOC(&l->gpu.grads, l->out_n * net->batch_size * sizeof(float));
		CUDA_MALLOC((void**)(&l->gpu.maxpool_addresses), l->out_n * net->batch_size * sizeof(float*));
	}

	if (net->use_gpu) {
		l->forward = forward_maxpool_gpu;
		l->backward = backward_maxpool_gpu;
	}
	else {
		//l->forward = (l->ksize != 2 || l->stride != 2) ? forward_maxpool_general : forward_maxpool;
		l->forward = forward_maxpool_general;
		l->backward = backward_maxpool;
	}
	l->update = update_none;
}

void build_residual_layer(int i, network* net) {
	layer* l = &(net->layers[i]);
	layer* ls = net->layers;
	l->id = i;

	// Set default in_ids if none specified.
	if (l->in_ids.n == 0) {
		l->in_ids.a = (int*)xcalloc(1, sizeof(int));
		l->in_ids.a[0] = i - 1;
		l->in_ids.n = 1;
	}
	else {
		for (int j = 0; j < l->in_ids.n; j++) {
			if (l->in_ids.a[j] < 0) l->in_ids.a[j] += i;
			if (l->in_ids.a[j] > i || l->in_ids.a[j] < 0) {
				printf("Invalid in_id of %d for layer %d\n", l->in_ids.a[j], i);
				wait_for_key_then_exit();
			}
		}
	}

	// Build array of input layer addresses.
	l->in_layers = (layer**)xcalloc(l->in_ids.n, sizeof(layer*));
	if (i > 0) {
		for (size_t j = 0; j < l->in_ids.n; j++) {
			l->in_layers[j] = &ls[l->in_ids.a[j]];
		}
	}
	else { // if first layer
		l->in_layers[0] = net->input;
	}

	// Calculate input dimensions.
	l->w = l->in_layers[0]->out_w;
	l->h = l->in_layers[0]->out_h;
	l->c = l->in_layers[0]->out_c;
	l->n = l->w * l->h * l->c;

	for (size_t j = 1; j < l->in_ids.n; j++) {
		if (!(l->in_layers[j]->out_w == l->w && l->in_layers[j]->out_h == l->h && l->in_layers[j]->out_c == l->c)) {
			printf("Inputs to residual layers must have matching dimensions.\n"
				"Residual layer %d, Input layer %d\n", i, l->in_layers[j]->id);
			wait_for_key_then_exit();
		}
	}

	// Calculate output dimensions.
	l->out_w = l->w;
	l->out_h = l->h;
	l->out_c = l->c;
	l->out_n = l->out_w * l->out_h * l->out_c;

	l->Z = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	l->act_inputs = l->Z;
	l->grads = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.Z, l->out_n * net->batch_size * sizeof(float));
		l->gpu.act_inputs = l->gpu.Z;
		CUDA_MALLOC(&l->gpu.grads, l->out_n * net->batch_size * sizeof(float));
	}

	if (l->activation) {
		l->output = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
		if (net->use_gpu) CUDA_MALLOC(&l->gpu.output, l->out_n * net->batch_size * sizeof(float));
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

void build_route_layer(int i, network* net) {
	layer* l = &(net->layers[i]);
	layer* ls = net->layers;
	l->id = i;

	// Set default in_ids if none specified.
	if (l->in_ids.n == 0) {
		l->in_ids.a = (int*)xcalloc(1, sizeof(int));
		l->in_ids.a[0] = i - 1;
		l->in_ids.n = 1;
	}
	else {
		for (int j = 0; j < l->in_ids.n; j++) {
			if (l->in_ids.a[j] < 0) l->in_ids.a[j] += i;
			if (l->in_ids.a[j] > i || l->in_ids.a[j] < 0) {
				printf("Invalid in_id of %d for layer %d\n", l->in_ids.a[j], i);
				wait_for_key_then_exit();
			}
		}
	}

	// Build array of input layer addresses.
	l->in_layers = (layer**)xcalloc(l->in_ids.n, sizeof(layer*));
	if (i > 0) {
		for (size_t j = 0; j < l->in_ids.n; j++) {
			l->in_layers[j] = &ls[l->in_ids.a[j]];
		}
	}
	else { // if first layer
		l->in_layers[0] = net->input;
	}

	// Calculate input dimensions.
	l->w = l->in_layers[0]->out_w;
	l->h = l->in_layers[0]->out_h;
	l->c = l->in_layers[0]->out_c;
	l->n = l->w * l->h * l->c;

	for (size_t j = 1; j < l->in_ids.n; j++) {
		if (!(l->in_layers[j]->out_w == l->w && l->in_layers[j]->out_h == l->h && l->in_layers[j]->out_c == l->c)) {
			printf("Inputs smust have matching width and height.\n"
				"Route layer %d, Input layer %d\n", i, l->in_layers[j]->id);
			wait_for_key_then_exit();
		}
	}

	// Calculate output dimensions.
	l->out_w = l->w;
	l->out_h = l->h;
	l->out_c = l->c;
	l->out_n = l->out_w * l->out_h * l->out_c;

	l->Z = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	l->act_inputs = l->Z;
	l->grads = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.Z, l->out_n * net->batch_size * sizeof(float));
		l->gpu.act_inputs = l->gpu.Z;
		CUDA_MALLOC(&l->gpu.grads, l->out_n * net->batch_size * sizeof(float));
	}

	if (l->activation) {
		l->output = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
		if (net->use_gpu) CUDA_MALLOC(&l->gpu.output, l->out_n * net->batch_size * sizeof(float));
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

void build_avgpool_layer(int i, network* net) {
	layer* l = &(net->layers[i]);
	layer* ls = net->layers;
	l->id = i;

	// Set default in_ids if none specified.
	if (l->in_ids.n == 0) {
		l->in_ids.a = (int*)xcalloc(1, sizeof(int));
		l->in_ids.a[0] = i - 1;
		l->in_ids.n = 1;
	}
	else {
		for (int j = 0; j < l->in_ids.n; j++) {
			if (l->in_ids.a[j] < 0) l->in_ids.a[j] += i;
			if (l->in_ids.a[j] > i || l->in_ids.a[j] < 0) {
				printf("Invalid in_id of %d for layer %d\n", l->in_ids.a[j], i);
				wait_for_key_then_exit();
			}
		}
	}

	// Build array of input layer addresses.
	l->in_layers = (layer**)xcalloc(l->in_ids.n, sizeof(layer*));
	if (i > 0) {
		for (size_t j = 0; j < l->in_ids.n; j++) {
			l->in_layers[j] = &ls[l->in_ids.a[j]];
		}
	}
	else { // if first layer
		l->in_layers[0] = net->input;
	}

	// Calculate input dimensions.
	l->w = l->in_layers[0]->out_w;
	l->h = l->in_layers[0]->out_h;
	l->c = l->in_layers[0]->out_c;
	for (size_t j = 1; j < l->in_ids.n; j++) {
		layer* inl = l->in_layers[j];
		if (l->w != inl->out_w || l->h != inl->out_h) {
			printf("Invalid input layer dimensions. Width and height must match.\n");
			printf("Layer %d %zux%zu, Input layer %d %zux%zu\n", i, l->w, l->h, inl->id, inl->out_w, inl->out_h);
			wait_for_key_then_exit();
		}
		l->c += inl->out_c;
	}
	l->n = l->w * l->h * l->c;

	// Calculate output dimensions.
	l->out_w = 1;
	l->out_h = 1;
	l->out_c = l->c;
	l->out_n = l->out_w * l->out_h * l->out_c;

	l->Z = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	l->act_inputs = l->Z;
	l->grads = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.Z, l->out_n * net->batch_size * sizeof(float));
		l->gpu.act_inputs = l->gpu.Z;
		CUDA_MALLOC(&l->gpu.grads, l->out_n * net->batch_size * sizeof(float));
	}

	if (l->activation) {
		l->output = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
		if (net->use_gpu) {
			CUDA_MALLOC(&l->gpu.output, l->out_n * net->batch_size * sizeof(float));
		}
	}
	else {
		l->output = l->Z;
		l->gpu.output = l->gpu.Z;
	}

	if (net->use_gpu) {
		l->forward = forward_avgpool_gpu;
		l->backward = backward_avgpool_gpu;
	}
	else {
		l->forward = forward_avgpool;
		l->backward = backward_avgpool;
	}
	l->update = update_none;

	set_activate(l, net->use_gpu);
}

void build_upsample_layer(int i, network* net) {
	layer* l = &(net->layers[i]);
	layer* ls = net->layers;
	l->id = i;

	// Set default in_ids if none specified.
	if (l->in_ids.n == 0) {
		l->in_ids.a = (int*)xcalloc(1, sizeof(int));
		l->in_ids.a[0] = i - 1;
		l->in_ids.n = 1;
	}
	else {
		for (int j = 0; j < l->in_ids.n; j++) {
			if (l->in_ids.a[j] < 0) l->in_ids.a[j] += i;
			if (l->in_ids.a[j] > i || l->in_ids.a[j] < 0) {
				printf("Invalid in_id of %d for layer %d\n", l->in_ids.a[j], i);
				wait_for_key_then_exit();
			}
		}
	}

	// Build array of input layer addresses.
	l->in_layers = (layer**)xcalloc(l->in_ids.n, sizeof(layer*));
	if (i > 0) {
		for (size_t j = 0; j < l->in_ids.n; j++) {
			l->in_layers[j] = &ls[l->in_ids.a[j]];
		}
	}

	// Calculate input dimensions.
	l->w = l->in_layers[0]->out_w;
	l->h = l->in_layers[0]->out_h;
	l->c = l->in_layers[0]->out_c;
	for (size_t j = 1; j < l->in_ids.n; j++) {
		layer* inl = l->in_layers[j];
		if (l->w != inl->out_w || l->h != inl->out_h) {
			printf("Invalid input layer dimensions. Width and height must match.\n");
			printf("Layer %d %zux%zu, Input layer %d %zux%zu\n", i, l->w, l->h, inl->id, inl->out_w, inl->out_h);
			wait_for_key_then_exit();
		}
		l->c += inl->out_c;
	}
	l->n = l->w * l->h * l->c;

	// Calculate output dimensions.
	l->out_w = l->w * l->ksize;
	l->out_h = l->h * l->ksize;
	l->out_c = l->c;
	l->out_n = l->out_w * l->out_h * l->out_c;
	l->Z = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	if (net->use_gpu) {
		CUDA_MALLOC(&l->gpu.Z, l->out_n * net->batch_size * sizeof(float));
	}

	l->act_inputs = l->Z;
	l->gpu.act_inputs = l->gpu.Z;

	if (l->activation) {
		l->output = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
		if (net->use_gpu) CUDA_MALLOC(&l->gpu.output, l->out_n * net->batch_size * sizeof(float));
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

void build_detect_layer(int i, network* net) {
	if (net->type != NET_DETECT) {
		printf("Invalid network cfg: Cannot have a detect layer in a non-detector network.\n");
		wait_for_key_then_exit();
	}

	layer* l = &(net->layers[i]);
	layer* ls = net->layers;
	l->id = i;

	if (l->n_classes == 0) l->n_classes = net->n_classes;

	// Set default in_ids if none specified
	if (l->in_ids.n == 0) {
		l->in_ids.a = (int*)xcalloc(1, sizeof(int));
		l->in_ids.a[0] = i - 1;
		l->in_ids.n = 1;
	}
	else {
		for (int j = 0; j < l->in_ids.n; j++) {
			if (l->in_ids.a[j] < 0) l->in_ids.a[j] += i;
			if (l->in_ids.a[j] > i || l->in_ids.a[j] < 0) {
				printf("Invalid in_id of %d for layer %d\n", l->in_ids.a[j], i);
				wait_for_key_then_exit();
			}
		}
	}

	// Build array of input layer addresses.
	l->in_layers = (layer**)xcalloc(l->in_ids.n, sizeof(layer*));
	if (i > 0) {
		for (size_t j = 0; j < l->in_ids.n; j++) {
			l->in_layers[j] = &ls[l->in_ids.a[j]];
		}
	}
	else { // if first layer
		l->in_layers[0] = net->input;
	}
	if (l->in_layers[0]->activation != ACT_SIGMOID) {
		printf("Invalid cfg: The input layer to a detect layer must have an sigmoid activation.\n");
		wait_for_key_then_exit();
	}

	// Calculate input dimensions.
	l->w = l->in_layers[0]->out_w;
	l->h = l->in_layers[0]->out_h;
	l->c = l->in_layers[0]->out_c;
	for (size_t j = 1; j < l->in_ids.n; j++) {
		layer* inl = l->in_layers[j];
		if (l->w != inl->out_w || l->h != inl->out_h) {
			printf("Invalid input layer dimensions. Width and height must match.\n");
			printf("Layer %d %zux%zu, Input layer %d %zux%zu\n", i, l->w, l->h, inl->id, inl->out_w, inl->out_h);
			wait_for_key_then_exit();
		}
		l->c += inl->out_c;
	}
	size_t c = (NUM_ANCHOR_PARAMS + l->n_classes) * l->n_anchors;
	if (l->c != c) {
		printf("Depth mismatch between Detect layer and its input layers. (%zu =/= %zu)\n", c, l->c);
		wait_for_key_then_exit();
	}
	l->n = l->w * l->h * l->c;

	l->grads = (float*)xcalloc(l->n * net->batch_size, sizeof(float));
	l->errors = (float*)xcalloc(l->n * net->batch_size, sizeof(float));

	l->out_w = l->w;
	l->out_h = l->h;
	l->out_c = l->c;

	l->forward = forward_detect;
	l->backward = backward_detect;
	l->update = update_none;

	l->activation = ACT_NONE;

	l->output = (float*)xcalloc(l->n * net->batch_size, sizeof(float));

	// set anchors
	if (net->w != net->h || l->out_w != l->out_h) {
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
	l->detections = (bbox*)xcalloc(l->w * l->h * l->n_anchors, sizeof(bbox));
	l->sorted = (bbox**)xcalloc(l->w * l->h * l->n_anchors, sizeof(bbox*));
	// TODO: Make these cfg parameters
	l->nms_obj_thresh = 0.5F;
	l->nms_cls_thresh = 0.5F;
	l->nms_iou_thresh = 0.5F;
	l->ignore_thresh = 0.7F;
	l->iou_thresh = 0.2F;
	l->obj_normalizer = 0.4F;
	l->max_box_grad = 2.0F;
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

void get_activation_grads(layer* l, size_t batch_size) {
	if (l->activation == ACT_MISH) get_grads_mish(l->grads, l->act_inputs, l->out_n, batch_size);
	else if (l->activation == ACT_RELU) get_grads_relu(l->grads, l->act_inputs, l->out_n, batch_size);
	else if (l->activation == ACT_LEAKY) get_grads_leaky_relu(l->grads, l->act_inputs, l->out_n, batch_size);
	else if (l->activation == ACT_SIGMOID) get_grads_sigmoid(l->grads, l->output, l->out_n, batch_size);
	else if (l->activation == ACT_SOFTMAX);
	else if (l->activation == ACT_TANH) get_grads_tanh(l->grads, l->act_inputs, l->out_n, batch_size);
	else if (l->activation == ACT_NONE);
	else {
		printf("Invalid activation function.");
		wait_for_key_then_exit();
	}
}

#pragma warning (suppress:4100)
void get_activation_grads_gpu(layer* l, size_t batch_size) {
#ifdef GPU
	if (l->activation == ACT_MISH) get_grads_mish_gpu(l->gpu.grads, l->gpu.act_inputs, (int)l->out_n, (int)batch_size);
	else if (l->activation == ACT_RELU) get_grads_relu_gpu(l->gpu.grads, l->gpu.act_inputs, (int)l->out_n, (int)batch_size);
	else if (l->activation == ACT_LEAKY) get_grads_leaky_relu_gpu(l->gpu.grads, l->gpu.act_inputs, (int)l->out_n, (int)batch_size);
	else if (l->activation == ACT_SIGMOID) get_grads_sigmoid_gpu(l->gpu.grads, l->gpu.output, (int)l->out_n, (int)batch_size);
	else if (l->activation == ACT_SOFTMAX);
	else if (l->activation == ACT_TANH) get_grads_tanh_gpu(l->gpu.grads, l->gpu.act_inputs, (int)l->out_n, (int)batch_size);
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
		CUDA_MALLOC(&net->gpu.reg_loss, sizeof(float));
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
		xfree(n->class_names[i]);
	}
	xfree(n->class_names);
	xfree(n->step_percents.a);
	xfree(n->step_scaling.a);
	free_layer_members(n->input);
	xfree(n->input);
	free_network_layers(n);
	xfree(n->workspace);
	xfree(n->dataset_dir);
	xfree(n->weights_file);
	xfree(n->save_dir);
	free_classifier_dataset_members(&n->data.clsr);
	xfree(n);
}

void free_network_layers(network* net) {
	for (size_t i = 0; i < net->n_layers; i++) {
		free_layer_members(&net->layers[i]);
	}
	xfree(net->layers);
}

void free_layer_members(layer* l) {
	xfree(l->output);
	xfree(l->weights);
	xfree(l->biases);
	xfree(l->weight_grads);
	xfree(l->bias_grads);
	xfree(l->Z);
	xfree(l->truth);
	xfree(l->means);
	xfree(l->variances);
	xfree(l->errors);
	xfree(l->in_ids.a);
	xfree(l->in_layers);
	if (l->type == LAYER_DETECT) {
		xfree(l->anchors);
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
	N += l->n_filters;  // # of biases
	if (l->batchnorm) N += l->out_c * 3;  // gammas, rolling means & variances
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
	else if (l->type == LAYER_FC) print_layer_conv(l);
	else if (l->type == LAYER_RESIDUAL) print_layer_residual(l);
	else printf("NONE_LAYER\n");
}

void print_layer_conv(layer* l) {
	printf("\n[LAYER]\n");
	printf("id: %d\n", l->id);
	printf("layer_type: ");
	print_layertype(l->type);
	printf("activation: ");
	print_activation(l->activation);
	printf("batchnorm: %d\n", l->batchnorm);
	printf("w, h, c: %zu, %zu, %zu\n", l->w, l->h, l->c);
	printf("n_filters: %zu\n", l->n_filters);
	printf("ksize: %zu\n", l->ksize);
	printf("stride: %zu\n", l->stride);
	printf("pad: %zu\n", l->pad);
	printf("# of inputs: %zu\n", l->n);
	printf("# of outputs: %zu\n", l->out_n);
	printf("# of weights: %zu\n", l->n_weights);
	printf("train: %d\n", l->train);
	printf("in_ids: ");
	print_intarr(&(l->in_ids));
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
	printf("w, h, c: %zu, %zu, %zu\n", l->w, l->h, l->c);
	printf("stride: %zu\n", l->stride);
	printf("# of inputs: %zu\n", l->n);
	printf("# of outputs: %zu\n", l->out_n);
	printf("# of weights: %zu\n", l->n_weights);
	printf("train: %d\n", l->train);
	printf("in_ids: ");
	print_intarr(&(l->in_ids));
	printf("[END LAYER]\n");
}

void print_layer_maxpool(layer* l) {
	printf("\n[LAYER]\n");
	printf("id: %d\n", l->id);
	printf("layer_type: ");
	print_layertype(l->type);
	printf("w, h, c: %zu, %zu, %zu\n", l->w, l->h, l->c);
	printf("ksize: %zu\n", l->ksize);
	printf("stride: %zu\n", l->stride);
	printf("pad: %zu\n", l->pad);
	printf("# of inputs: %zu\n", l->n);
	printf("# of outputs: %zu\n", l->out_n);
	printf("train: %d\n", l->train);
	printf("in_ids: ");
	print_intarr(&(l->in_ids));
	printf("[END LAYER]\n");
}

void print_layer_residual(layer* l) {
	printf("\n[LAYER]\n");
	printf("id: %d\n", l->id);
	printf("layer_type: ");
	print_layertype(l->type);
	printf("activation: ");
	print_activation(l->activation);
	printf("w, h, c: %zu, %zu, %zu\n", l->w, l->h, l->c);
	printf("# of inputs: %zu\n", l->n);
	printf("# of outputs: %zu\n", l->out_n);
	printf("in_ids: ");
	print_intarr(&(l->in_ids));
	printf("[END LAYER]\n");
}

void print_lrpolicy(LR_POLICY lrp) {
	if (lrp == LR_STEPS) printf("steps\n");
	else printf("NONE\n");
}

void print_layertype(LAYER_TYPE lt) {
	if (lt == LAYER_NONE) printf("none\n");
	else if (lt == LAYER_AVGPOOL) printf("avgpool\n");
	else if (lt == LAYER_CLASSIFY) printf("classify\n");
	else if (lt == LAYER_CONV) printf("conv\n");
	else if (lt == LAYER_DETECT) printf("detect\n");
	else if (lt == LAYER_FC) printf("fc\n");
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
		get_in_ids_str(idbuf, 50, l->in_ids);
		printf("[%d] %s inputs[%s] %zux%zux%zu -> %zux%zux%zu pad: %zu stride: %zu ksize: %zu\n", l->id, ltbuf, idbuf, l->w, l->h, l->c, l->out_w, l->out_h, l->out_c, l->pad, l->stride, l->ksize);
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
	else if (lt == LAYER_AVGPOOL) strcpy(buf, "avgpool");
	else if (lt == LAYER_CLASSIFY) strcpy(buf, "classify");
	else if (lt == LAYER_CONV) strcpy(buf, "conv");
	else if (lt == LAYER_DETECT) strcpy(buf, "detect");
	else if (lt == LAYER_FC) strcpy(buf, "fc");
	else if (lt == LAYER_MAXPOOL) strcpy(buf, "maxpool");
	else if (lt == LAYER_RESIDUAL) strcpy(buf, "residual");
	else if (lt == LAYER_UPSAMPLE) strcpy(buf, "upsample");
}

void get_in_ids_str(char* buf, size_t bufsize, intarr inputs) {
	if (bufsize < 50) {
		printf("Error: bufsize too small.\n");
		wait_for_key_then_exit();
	}
	memset(buf, 0, bufsize);
	if (!inputs.a) return;
	snprintf(buf, bufsize, "%d", inputs.a[0]);
	for (size_t i = 1; i < inputs.n; i++) {
		size_t length = strlen(buf);
		snprintf(&buf[length], bufsize - length, ",%d", inputs.a[i]);
	}
}