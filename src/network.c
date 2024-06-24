#include "network.h"
#include <assert.h>
#include <stdlib.h>
#include <omp.h>
#include "utils.h"
#include "xallocs.h"
#include "layer_conv.h"
#include "layer_classify.h"
#include "activations.h"
#include "costs.h"
#include "derivatives.h"


void build_input_layer(network* net);
void build_layer(int i, network* net);
void build_conv_layer(int i, network* net);
void build_classify_layer(int i, network* net);
void build_detect_layer(int i, network* net);
void set_activate(layer* l);
void set_cost(layer* l);
void free_network(network* net);
void free_layers(network* net);
void free_layer_members(layer* l);
void print_layer_conv(layer* l);
void print_layer_classify(layer* l);
void print_weights(network* net);


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
}

void build_layer(int i, network* net) {
	layer* l = &(net->layers[i]);
	assert(l->type != LAYER_NONE);
	if (l->type == LAYER_CONV) build_conv_layer(i, net);
	else if (l->type == LAYER_CLASSIFY) build_classify_layer(i, net);
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
	l->output = (float*)xcalloc(l->out_n, sizeof(float));
}

// i = layer index in net->layers
void build_conv_layer(int i, network* net) {
	layer* l = &(net->layers[i]);
	layer* ls = net->layers;
	assert(l->id == i);

	// Set default in_ids and out_ids if none specified.
	if (l->in_ids.n == 0) {
		l->in_ids.a = (int*)xcalloc(1, sizeof(int));
		l->in_ids.a[0] = i - 1;
		l->in_ids.n = 1;
	}
	if (l->out_ids.n == 0) {
		l->out_ids.a = (int*)xcalloc(1, sizeof(int));
		l->out_ids.a[0] = i + 1;
		l->out_ids.n = 1;
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
		layer* inlay = l->in_layers[j];
		assert(l->w == inlay->out_w);
		assert(l->h == inlay->out_h);
		l->c += inlay->out_c;
	}
	l->n = l->w * l->h * l->c;

	// Calculate output dimensions.
	l->out_w = ((l->w + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_h = ((l->h + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_c = l->n_filters;
	l->out_n = l->out_w * l->out_h * l->out_c;

	l->output = (float*)xcalloc(l->out_n, sizeof(float));
	l->weights.n = l->n_filters * l->ksize * l->ksize * l->c;
	l->weights.a = (float*)xcalloc(l->weights.n, sizeof(float));
	l->biases = (float*)xcalloc(l->n_filters, sizeof(float));
	l->act_input = (float*)xcalloc(l->out_n, sizeof(float));
	l->grads = (float*)xcalloc(l->out_n, sizeof(float));
	l->weight_updates = (float*)xcalloc(l->weights.n, sizeof(float));

	l->forward = forward_conv;
	l->backprop = backprop_conv;
	set_activate(l);
}

// i = layer index in net->layers
void build_classify_layer(int i, network* net) {
	layer* l = &(net->layers[i]);
	layer* ls = net->layers;
	assert(l->id == i);

	if (l->n_classes == 0) {
		l->n_classes = net->n_classes;
		l->n_filters = l->n_classes;
	}

	if (l->in_ids.n == 0) {
		l->in_ids.a = (int*)xcalloc(1, sizeof(int));
		l->in_ids.a[0] = i - 1;
		l->in_ids.n = 1;
	}

	l->in_layers = (layer**)xcalloc(l->in_ids.n, sizeof(layer*));
	for (size_t j = 0; j < l->in_ids.n; j++) {
		l->in_layers[j] = &ls[l->in_ids.a[j]];
	}

	l->w = l->in_layers[0]->out_w;
	l->h = l->in_layers[0]->out_h;
	l->c = l->in_layers[0]->out_c;
	for (size_t j = 1; j < l->in_ids.n; j++) {
		layer* inlay = l->in_layers[i];
		assert(l->w == inlay->out_w);
		assert(l->h == inlay->out_h);
		l->c += inlay->out_c;
	}
	l->n = l->w * l->h * l->c;

	l->pad = 0;
	l->stride = 1;
	assert(l->w == l->h);
	l->ksize = l->w;

	l->out_w = ((l->w + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_h = ((l->h + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_c = l->n_filters;
	l->out_n = l->out_w * l->out_h * l->out_c;

	l->output = (float*)xcalloc(l->out_n, sizeof(float));
	l->weights.n = l->n_filters * l->ksize * l->ksize * l->c;
	l->weights.a = (float*)xcalloc(l->weights.n, sizeof(float));
	l->biases = (float*)xcalloc(l->n_filters, sizeof(float));
	l->act_input = (float*)xcalloc(l->out_n, sizeof(float));
	l->grads = (float*)xcalloc(l->out_n, sizeof(float)); // IDK IF THIS IS RIGHT LENGTH
	l->truth = (float*)xcalloc(net->n_classes, sizeof(float));
	l->weight_updates = (float*)xcalloc(l->weights.n, sizeof(float));

	l->forward = forward_classify;
	l->backprop = backprop_classify;
	set_activate(l);
	set_cost(l);
}

void build_detect_layer(int i, network* net) {
	// UNDER CONSTRUCTION - COPY/PASTED BUILD_CONV_LAYER FOR NOW
	layer* l = &(net->layers[i]);
	layer* ls = net->layers;
	assert(l->id == i);

	// Set default in_ids and out_ids if none specified.
	if (l->in_ids.n == 0) {
		l->in_ids.a = (int*)xcalloc(1, sizeof(int));
		l->in_ids.a[0] = i - 1;
		l->in_ids.n = 1;
	}
	if (l->out_ids.n == 0) {
		l->out_ids.a = (int*)xcalloc(1, sizeof(int));
		l->out_ids.a[0] = i + 1;
		l->out_ids.n = 1;
	}
	l->w = ls[l->in_ids.a[0]].w;
	l->h = ls[l->in_ids.a[0]].h;

	// Build array of input layer addresses.
	l->in_layers = (layer**)xcalloc(l->in_ids.n, sizeof(layer*));
	for (size_t j = 0; j < l->in_ids.n; j++) {
		l->in_layers[j] = &ls[l->in_ids.a[j]];
	}

	// Calculate input dimensions.
	l->c = 0;
	for (size_t j = 0; j < l->in_ids.n; j++) {
		layer* in_layer = l->in_layers[j];
		assert(l->w == in_layer->out_w);
		assert(l->h == in_layer->out_h);
		l->c += in_layer->out_c;
	}
	l->n = l->w * l->h * l->c;

	// Calculate output dimensions.
	l->out_w = ((l->w + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_h = ((l->h + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_c = l->n_filters;
	l->out_n = l->out_w * l->out_h * l->out_c;

	l->output = (float*)xcalloc(l->out_n, sizeof(float));
	l->weights.n = l->n_filters * l->ksize * l->ksize * l->c;
	l->weights.a = (float*)xcalloc(l->weights.n, sizeof(float));
	l->biases = (float*)xcalloc(l->n_filters, sizeof(float));
	l->act_input = (float*)xcalloc(l->out_n, sizeof(float));

	l->truth = (float*)xcalloc((net->n_classes + NUM_ANCHOR_PARAMS) * net->n_anchors, sizeof(float));

	l->forward = forward_conv;
	l->backprop = backprop_conv;
	set_activate(l);
	if (l->cost_type > 0) {
		set_cost(l);
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
	case ACT_SOFTMAX:
		l->activate = activate_softmax;
		break;
	default:
		l->activate = activate_none;
	}
}

void set_cost(layer* l) {
	switch (l->cost_type) {
	case COST_BCE:
		l->get_cost = cost_bce;
		break;
	case COST_CCE:
		if (l->activation == ACT_SIGMOID) {
			l->get_cost = cost_sigmoid_cce;
		}
		else if (l->activation == ACT_SOFTMAX) {
			l->get_cost = cost_softmax_cce;
		}
		else {
			printf("Error: Invalid cost and activation combination.\n Layer id = %d\n", l->id);
			wait_for_key_then_exit();
		}
		break;
	case COST_MSE:
		l->get_cost = cost_mse;
		break;
	default:
		printf("No cost function set. Layer id = %d\n", l->id);
	}
}

void free_network(network* n) {
	xfree(n->step_percents.a);
	xfree(n->step_scaling.a);
	xfree(n->input->output);
	xfree(n->input);
	free_layers(n);
	xfree(n->output);
	xfree(n);
}

void free_layers(network* net) {
	for (size_t i = 0; i < net->n_layers; i++) {
		free_layer_members(&net->layers[i]);
	}
	xfree(net->layers);
}

// TODO: Update
void free_layer_members(layer* l) {
	xfree(l->output);
	xfree(l->weights.a);
	xfree(l->biases);
	xfree(l->grads);
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
	printf("batch_size: %zu\n", l->batch_size);
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
	printf("out_ids: ");
	print_intarr(&(l->out_ids));
	printf("[END LAYER]\n");
}

void print_layer_classify(layer* l) {
	printf("\n[LAYER]\n");
	printf("id: %d\n", l->id);
	printf("layer_type: ");
	print_layertype(l->type);
	printf("cost: ");
	print_cost_type(l->cost_type);
	printf("# of classes: %zu\n", l->n_classes);
	printf("batch_size: %zu\n", l->batch_size);
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

void print_lrpolicy(LR_POLICY lrp) {
	if (lrp == LR_STEPS) printf("steps\n");
	else printf("NONE\n");
}

void print_layertype(LAYER_TYPE lt) {
	if (lt == LAYER_CONV) printf("conv\n");
	else if (lt == LAYER_CLASSIFY) printf("classify\n");
	else printf("NONE\n");
}

void print_activation(ACTIVATION a) {
	if (a == ACT_RELU) printf("relu\n");
	else if (a == ACT_LEAKY) printf("leaky relu\n");
	else if (a == ACT_MISH) printf("mish\n");
	else if (a == ACT_SIGMOID) printf("sigmoid\n");
	else if (a == ACT_SOFTMAX) printf("softmax\n");
	else printf("NONE\n");
}

void print_cost_type(COST_TYPE c) {
	if (c == COST_MSE) printf("mse\n");
	else if (c == COST_BCE) printf("bce\n");
	else if (c == COST_CCE) printf("cce\n");
	else printf("NONE\n");
}

void print_weights(network* net) {
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
