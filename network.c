#include "network.h"
#include <assert.h>
#include <stdlib.h>
#include <omp.h>
#include "xallocs.h"
#include "layer_conv.h"
#include "layer_classify.h"


void build_first_layer(network* net);
void build_layer(int i, network* net);
void build_conv_layer(int i, network* net);
void build_classify_layer(int i, network* net);
void set_activate(layer* l);
void free_network(network* net);
void free_layers(network* net);
void free_layer_members(layer* l);
void print_layer_conv(layer* l);
void print_layer_classify(layer* l);


network* new_network(size_t num_of_layers) {
	network* net = (network*)xcalloc(1, sizeof(network));
	net->n_layers = num_of_layers;
	net->layers = (layer*)xcalloc(num_of_layers, sizeof(layer));
	return net;
}

void build_network(network* net) {
	for (int i = 0; i < net->n_layers; i++) {
		build_layer(i, net);
	}
}

void build_layer(int i, network* net) {
	layer* l = &(net->layers[i]);
	assert(l->type != NONE_LAYER);
	if (i == 0) build_first_layer(net);
	else if (l->type == CONV) build_conv_layer(i, net);
	else if (l->type == CLASSIFY) build_classify_layer(i, net);
}

void build_first_layer(network* net) {
	layer* l = &(net->layers[0]);
	l->w = net->w;
	l->h = net->h;
	l->c = net->c;
	l->n = l->w * l->h * l->c;
	l->out_w = ((l->w + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_h = ((l->h + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_c = l->n_filters;
	l->out_n = l->out_w * l->out_h * l->out_c;
	l->output = (float*)xcalloc(l->out_n, sizeof(float));
	l->weights.n = l->n_filters * l->ksize * l->ksize * l->c;
	l->weights.a = (float*)xcalloc(l->weights.n, sizeof(float));
	l->biases = (float*)xcalloc(l->n_filters, sizeof(float));
	l->act_input = (float*)xcalloc(l->out_n, sizeof(float));

	l->forward = forward_layer_first;
	l->backward = backward_layer_first;
	set_activate(l);
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

	l->forward = forward_layer_conv;
	l->backward = backward_layer_conv;
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
	if (l->cost == NONE_COST_TYPE) l->cost = net->cost;

	if (l->in_ids.n == 0) {
		l->in_ids.a = (int*)xcalloc(1, sizeof(int));
		l->in_ids.a[0] = i - 1;
		l->in_ids.n = 1;
	}
	l->w = ls[l->in_ids.a[0]].out_w;
	l->h = ls[l->in_ids.a[0]].out_h;
	l->c = ls[l->in_ids.a[0]].out_c;
	for (size_t j = 1; j < l->in_ids.n; j++) {
		l->c += ls[j].out_c;
		assert(l->w == ls[l->in_ids.a[j]].out_w);
		assert(l->h == ls[l->in_ids.a[j]].out_h);
	}
	l->n = l->w * l->h * l->c;

	l->in_layers = (layer**)xcalloc(l->in_ids.n, sizeof(layer*));
	for (size_t j = 0; j < l->in_ids.n; j++) {
		l->in_layers[j] = &ls[l->in_ids.a[j]];
	}

	l->out_w = 1;
	l->out_h = 1;
	l->out_c = l->n_filters;
	l->out_n = l->out_w * l->out_h * l->out_c;

	l->output = (float*)xcalloc(l->out_n, sizeof(float));
	l->weights.n = l->n_filters * l->n;
	l->weights.a = (float*)xcalloc(l->weights.n, sizeof(float));
	l->biases = (float*)xcalloc(l->n_filters, sizeof(float));
	l->act_input = (float*)xcalloc(l->out_n, sizeof(float));
	l->stride = 1;

	l->forward = forward_layer_classify;
	l->backward = backward_layer_classify;
	set_activate(l);
}

void set_activate(layer* l) {
	if (l->type == CONV) {
		switch (l->activation) {
		case RELU:
			l->activate = activate_conv_relu;
			break;
		case MISH:
			l->activate = activate_conv_mish;
			break;
		case LOGISTIC:
			l->activate = activate_conv_logistic;
			break;
		default:
			l->activate = activate_conv_none;
			break;
		}
		return;
	}
	else if (l->type == CLASSIFY) {
		l->activate = activate_classify;
		return;
	}
}

void free_network(network* n) {
	xfree(n->step_percents.a);
	xfree(n->step_scaling.a);
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

void free_layer_members(layer* l) {
	xfree(l->output);
	xfree(l->weights.a);
	xfree(l->biases);
	xfree(l->delta);
	xfree(l->means);
	xfree(l->variances);
	xfree(l->losses);
	xfree(l->in_ids.a);
	xfree(l->out_ids.a);
	xfree(l->in_layers);
	if (l->type == OBJ_DET) {
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
	if (l->type == CONV) print_layer_conv(l);
	else if (l->type == CLASSIFY) print_layer_classify(l);
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
	print_cost_type(l->cost);
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
	if (lt == CONV) printf("conv\n");
	else if (lt == CLASSIFY) printf("classify\n");
	else printf("NONE\n");
}

void print_activation(ACTIVATION a) {
	if (a == RELU) printf("relu\n");
	else if (a == MISH) printf("mish\n");
	else if (a == LOGISTIC) printf("logistic\n");
	else printf("NONE\n");
}

void print_cost_type(COST_TYPE c) {
	if (c == MSE) printf("mse\n");
	else if (c == BCE) printf("bce\n");
	else if (c == CCE) printf("cce\n");
	else printf("NONE\n");
}


