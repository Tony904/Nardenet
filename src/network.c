#include "network.h"
#include <assert.h>
#include <stdlib.h>
#include <omp.h>
#include "utils.h"
#include "xallocs.h"
#include "layer_conv.h"
#include "layer_classify.h"
#include "layer_maxpool.h"
#include "layer_residual.h"
#include "layer_detect.h"
#include "activations.h"
#include "loss.h"
#include "derivatives.h"


void build_input_layer(network* net);
void build_layer(int i, network* net);
void build_conv_layer(int i, network* net);
void build_fc_layer(int i, network* net);
void build_maxpool_layer(int i, network* net);
void build_residual_layer(int i, network* net);
void build_classify_layer(int i, network* net);
void build_detect_layer(int i, network* net);

void set_activate(layer* l);
void set_loss(layer* l);

size_t get_train_params_count(network* net);
size_t get_layer_param_count(layer* l);

void free_network(network* net);
void free_network_layers(network* net);
void free_layer_members(layer* l);

void print_layer_conv(layer* l);
void print_layer_classify(layer* l);
void print_layer_maxpool(layer* l);
void print_layer_residual(layer* l);


network* new_network(size_t num_of_layers) {
	network* net = (network*)xcalloc(1, sizeof(network));
	net->input = (layer*)xcalloc(1, sizeof(layer));
	net->n_layers = num_of_layers;
	net->layers = (layer*)xcalloc(num_of_layers, sizeof(layer));
	return net;
}

void build_network(network* net) {
	build_input_layer(net);
	size_t largest_workspace = 0;
	layer* ls = net->layers;
	for (int i = 0; i < net->n_layers; i++) {
		build_layer(i, net);
		layer* l = &ls[i];
		size_t wssize = l->out_w * l->out_h * l->ksize * l->ksize * l->c;
		largest_workspace = max(largest_workspace, wssize);
	}
	net->workspace.a = (float*)xcalloc(largest_workspace, sizeof(float));
	net->workspace.n = largest_workspace;
	net->current_learning_rate = net->learning_rate;
	if (net->regularization == REG_L1) {
		net->reg_loss = loss_l1;
		net->regularize_weights = regularize_l1;
	}
	else if (net->regularization == REG_L2) {
		net->reg_loss = loss_l2;
		net->regularize_weights = regularize_l2;
	}
	else {
		net->regularize_weights = regularize_none;
	}
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
}

/* i = layer index in net->layers */
void build_conv_layer(int i, network* net) {
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
			printf("Invalid input layer dimensions. Width and height must match.\n Layer %d, Input layer %d.\n", i, inl->id);
		}
		l->c += inl->out_c;
	}
	l->n = l->w * l->h * l->c;

	// Calculate output dimensions.
	l->out_w = ((l->w + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_h = ((l->h + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_c = l->n_filters;
	l->out_n = l->out_w * l->out_h * l->out_c;
	l->Z = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	l->weights.n = l->n_filters * l->ksize * l->ksize * l->c;
	l->weights.a = (float*)xcalloc(l->weights.n, sizeof(float));
	l->biases = (float*)xcalloc(l->n_filters, sizeof(float));
	l->grads = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	l->weight_grads = (float*)xcalloc(l->weights.n, sizeof(float));
	l->bias_grads = (float*)xcalloc(l->n_filters, sizeof(float));
	l->weights_velocity = (float*)xcalloc(l->weights.n, sizeof(float));
	l->biases_velocity = (float*)xcalloc(l->n_filters, sizeof(float));
	if (l->batch_norm) {
		l->Z_norm = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
		l->act_inputs = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
		l->means = (float*)xcalloc(l->out_c, sizeof(float));
		l->variances = (float*)xcalloc(l->out_c, sizeof(float));
		l->gammas = (float*)xcalloc(l->out_c, sizeof(float));
		l->gamma_grads = (float*)xcalloc(l->out_c, sizeof(float));
		l->gammas_velocity = (float*)xcalloc(l->out_c, sizeof(float));
		fill_array(l->gammas, l->out_c, 1.0F);
		l->rolling_means = (float*)xcalloc(l->out_c, sizeof(float));
		l->rolling_variances = (float*)xcalloc(l->out_c, sizeof(float));
	}
	else {
		l->act_inputs = l->Z;
	}

	if (l->activation) l->output = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	else l->output = l->act_inputs;

	l->forward = forward_conv;
	l->backward = backward_conv;
	l->update = update_conv;
	
	set_activate(l);
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
			printf("Invalid input layer dimensions. Width and height must match.\n Layer %d, Input layer %d.\n", i, inl->id);
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
	l->weights.n = l->n_filters * l->ksize * l->ksize * l->c;
	l->weights.a = (float*)xcalloc(l->weights.n, sizeof(float));
	l->biases = (float*)xcalloc(l->n_filters, sizeof(float));
	l->grads = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	l->weight_grads = (float*)xcalloc(l->weights.n, sizeof(float));
	l->bias_grads = (float*)xcalloc(l->n_filters, sizeof(float));
	l->weights_velocity = (float*)xcalloc(l->weights.n, sizeof(float));
	l->biases_velocity = (float*)xcalloc(l->n_filters, sizeof(float));
	if (l->batch_norm) {
		l->Z_norm = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
		l->act_inputs = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
		l->means = (float*)xcalloc(l->out_c, sizeof(float));
		l->variances = (float*)xcalloc(l->out_c, sizeof(float));
		l->gammas = (float*)xcalloc(l->out_c, sizeof(float));
		l->gamma_grads = (float*)xcalloc(l->out_c, sizeof(float));
		l->gammas_velocity = (float*)xcalloc(l->out_c, sizeof(float));
		fill_array(l->gammas, l->out_c, 1.0F);
		l->rolling_means = (float*)xcalloc(l->out_c, sizeof(float));
		l->rolling_variances = (float*)xcalloc(l->out_c, sizeof(float));
	}
	else {
		l->act_inputs = l->Z;
	}

	if (l->activation) l->output = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	else l->output = l->act_inputs;

	l->forward = forward_conv;
	l->backward = backward_conv;
	l->update = update_conv;
	set_activate(l);
}

// i = layer index in net->layers
void build_classify_layer(int i, network* net) {
	layer* l = &(net->layers[i]);
	layer* ls = net->layers;
	l->id = i;

	if (l->n_classes == 0) l->n_classes = net->n_classes;
	l->n_filters = l->n_classes;

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
			printf("Invalid input layer dimensions. Width and height must match.\n Layer %d, Input layer %d.\n", i, inl->id);
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
	l->weights.n = l->n_filters * l->ksize * l->ksize * l->c;
	l->weights.a = (float*)xcalloc(l->weights.n, sizeof(float));
	l->biases = (float*)xcalloc(l->n_filters, sizeof(float));
	l->grads = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	l->weight_grads = (float*)xcalloc(l->weights.n, sizeof(float));
	l->bias_grads = (float*)xcalloc(l->n_filters, sizeof(float));
	l->weights_velocity = (float*)xcalloc(l->weights.n, sizeof(float));
	l->biases_velocity = (float*)xcalloc(l->n_filters, sizeof(float));
	l->truth = (float*)xcalloc(net->n_classes * net->batch_size, sizeof(float));
	l->errors = (float*)xcalloc(net->n_classes * net->batch_size, sizeof(float));
	if (l->batch_norm) {
		l->Z_norm = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
		l->act_inputs = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
		l->means = (float*)xcalloc(l->out_c, sizeof(float));
		l->variances = (float*)xcalloc(l->out_c, sizeof(float));
		l->gammas = (float*)xcalloc(l->out_c, sizeof(float));
		l->gamma_grads = (float*)xcalloc(l->out_c, sizeof(float));
		l->gammas_velocity = (float*)xcalloc(l->out_c, sizeof(float));
		fill_array(l->gammas, l->out_c, 1.0F);
		l->rolling_means = (float*)xcalloc(l->out_c, sizeof(float));
		l->rolling_variances = (float*)xcalloc(l->out_c, sizeof(float));
	}
	else {
		l->act_inputs = l->Z;
	}
	
	if (l->activation) l->output = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	else l->output = l->act_inputs;
	
	l->forward = forward_classify;
	l->backward = backward_classify;
	l->update = update_classify;

	set_activate(l);

	set_loss(l);
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
			printf("Invalid input layer dimensions. Width and height must match.\n Layer %d, Input layer %d.\n", i, inl->id);
		}
		l->c += inl->out_c;
	}
	l->n = l->w * l->h * l->c;

	// Calculate output dimensions.
	l->out_w = ((l->w + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_h = ((l->h + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_c = l->c;
	l->out_n = l->out_w * l->out_h * l->out_c;

	l->output = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	l->grads = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	l->maxpool_addresses = (float**)xcalloc(l->out_n * net->batch_size, sizeof(float*));

	l->forward = (l->ksize != 2 || l->stride != 2) ? forward_maxpool_general : forward_maxpool;
	l->backward = backward_maxpool;
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
		if (!(l->in_layers[j]->w == l->w && l->in_layers[j]->h == l->h && l->in_layers[j]->c == l->c)) {
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

	if (l->activation) l->output = (float*)xcalloc(l->out_n * net->batch_size, sizeof(float));
	else l->output = l->Z;

	l->forward = forward_residual;
	l->backward = backward_residual;
	l->update = update_none;

	set_activate(l);
}

void build_detect_layer(int i, network* net) {
	layer* l = &(net->layers[i]);
	layer* ls = net->layers;
	l->id = i;

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
			printf("Invalid input layer dimensions. Width and height must match.\n Layer %d, Input layer %d.\n", i, inl->id);
		}
		l->c += inl->out_c;
	}
	size_t c = (l->n_classes + NUM_ANCHOR_PARAMS) * l->n_anchors;
	if (l->c != c) {
		printf("Depth mismatch between Detect layer and its input layers. (%zu =/= %zu)\n", l->c, c);
	}
	l->n = l->w * l->h * l->c;

	l->forward = forward_detect;
	l->backward = backward_detect;
	l->update = update_none;

	// set anchors
	if (net->w != net->h || l->out_w != l->out_h) {
		printf("Network input and output width & height must be square.\n");
		wait_for_key_then_exit();
	}
	float cell_size = (float)l->out_w / (float)net->w;  // percentage of image width
	for (size_t j = 0; j < l->n_anchors; j++) {
		float w = l->anchors[j].w;  // percentage of image width
		float h = l->anchors[j].h;  // percentage of image height
		l->anchors[j].area = w * h;
		l->anchors[j].cx = 0.5F;  // percentage of cell size
		l->anchors[j].cy = 0.5F;
		// edges are offsets from top-left of cell (same as if top-left of cell was at (0, 0) of image)
		l->anchors[j].left = 0.5 * (cell_size - w);
		l->anchors[j].right = l->anchors[j].left + w;
		l->anchors[j].top = 0.5 * (cell_size - h);
		l->anchors[j].bottom = l->anchors[j].top + h;
	}
	
	det_cell* cells = (det_cell*)xcalloc(l->out_w * l->out_h, sizeof(det_cell));
	for (size_t row = 0; row < l->out_h; row++) {
		float cell_top = cell_size * (float)row;
		for (size_t col = 0; col < l->out_w; col++) {
			float cell_bottom = cell_size * (float)col;
			size_t cell_index = row * l->out_h + col;
			for (size_t j = 0; j < l->n_anchors; j++) {
				cells->anchors = (bbox*)xcalloc(l->n_anchors, sizeof(bbox));

				bbox* anchor = &anchors[cell + j * l->out_w * l->out_h];
				anchor->left = cell_cx - l->anchors[j].w * 0.5F;
				anchor->right = l->anchors[j].left + l->anchors[j].w;
				anchor->top = cell_cy - l->anchors[j].h * 0.5F;
				anchor->bottom = l->anchors[j].top + l->anchors[j].h;
				anchor->cx = cell_cx;
				anchor->cy = cell_cy;
				anchor->area = l->anchors[j].w * l->anchors[j].h;
			}
		}
	}
}

void set_activate(layer* l) {
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

void set_loss(layer* l) {
	switch (l->loss_type) {
	case LOSS_MAE:
		l->get_loss = loss_mae;
		break;
	case LOSS_CCE:
		if (l->activation == ACT_SIGMOID) {
			l->get_loss = loss_sigmoid_cce;
		}
		else if (l->activation == ACT_SOFTMAX) {
			l->get_loss = loss_softmax_cce;
		}
		else {
			printf("Error: Invalid loss and activation combination.\n Layer %d\n", l->id);
			wait_for_key_then_exit();
		}
		break;
	case LOSS_MSE:
		l->get_loss = loss_mse;
		break;
	default:
		printf("No loss function set. Layer %d\n", l->id);
	}
}

#pragma warning(suppress:4100)  // unreferenced formal parameter: 'net'
void update_none(layer* l, network* net) {
	l;
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
	xfree(n->workspace.a);
	xfree(n->dataset_dir);
	xfree(n->weights_file);
	xfree(n->backup_dir);
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
	xfree(l->weights.a);
	xfree(l->biases);
	xfree(l->weight_grads);
	xfree(l->bias_grads);
	xfree(l->Z);
	xfree(l->truth);
	xfree(l->means);
	xfree(l->variances);
	xfree(l->errors);
	xfree(l->in_ids.a);
	xfree(l->out_ids.a);
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
	size_t N = l->weights.n;
	N += l->n_filters;  // # of biases
	if (l->batch_norm) N += l->out_c * 3;  // gammas, rolling means & variances
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
	printf("batch_norm: %d\n", l->batch_norm);
	printf("w, h, c: %zu, %zu, %zu\n", l->w, l->h, l->c);
	printf("n_filters: %zu\n", l->n_filters);
	printf("ksize: %zu\n", l->ksize);
	printf("stride: %zu\n", l->stride);
	printf("pad: %zu\n", l->pad);
	printf("# of inputs: %zu\n", l->n);
	printf("# of outputs: %zu\n", l->out_n);
	printf("# of weights: %zu\n", l->weights.n);
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
	printf("# of weights: %zu\n", l->weights.n);
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
	printf("out_ids: ");
	print_intarr(&(l->out_ids));
	printf("[END LAYER]\n");
}

void print_lrpolicy(LR_POLICY lrp) {
	if (lrp == LR_STEPS) printf("steps\n");
	else printf("NONE\n");
}

void print_layertype(LAYER_TYPE lt) {
	if (lt == LAYER_CONV) printf("conv\n");
	else if (lt == LAYER_CLASSIFY) printf("classify\n");
	else if (lt == LAYER_MAXPOOL) printf("maxpool\n");
	else if (lt == LAYER_FC) printf("fc\n");
	else if (lt == LAYER_RESIDUAL) printf("residual\n");
	else printf("NONE\n");
}

void print_activation(ACTIVATION a) {
	if (a == ACT_RELU) printf("relu\n");
	else if (a == ACT_LEAKY) printf("leaky relu\n");
	else if (a == ACT_MISH) printf("mish\n");
	else if (a == ACT_SIGMOID) printf("sigmoid\n");
	else if (a == ACT_SOFTMAX) printf("softmax\n");
	else if (a == ACT_TANH) printf("tanh\n");
	else printf("NONE\n");
}

void print_loss_type(LOSS_TYPE c) {
	if (c == LOSS_MSE) printf("mse\n");
	else if (c == LOSS_MAE) printf("mae\n");
	else if (c == LOSS_CCE) printf("cce\n");
	else printf("NONE\n");
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
		n = l->weights.n;
		for (size_t j = 0; j < n; j++) {
			printf("%f, ", l->weights.a[j]);
			if ((j + 1) % 10 == 0) printf("\n");
		}
	}
}

void print_some_weights(layer* l, size_t n) {
	float* weights = l->weights.a;
	assert(l->weights.n >= n);
	for (size_t i = 0; i < n; i++) {
		printf("%f\n", weights[i]);
	}
}

void print_top_class_name(float* probs, size_t n_classes, char** class_names, int include_prob, int new_line) {
	size_t c = 0;
	float highscore = 0;
	for (size_t i = 0; i < n_classes; i++) {
		float p = probs[i];
		if (p > highscore) {
			highscore = p;
			c = i;
		}
	}
	if (new_line) {
		if (include_prob) printf("%s (%f)\n", class_names[c], highscore);
		else printf("%s\n", class_names[c]);
	}
	else {
		if (include_prob) printf("%s (%f)", class_names[c], highscore);
		else printf("%s", class_names[c]);
	}
}

void print_network_summary(network* net, int print_training_params) {
	printf("[NETWORK]\n"
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