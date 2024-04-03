#include "network.h"
#include "xallocs.h"
#include <assert.h>
#include <stdlib.h>


void build_first_layer(network* net);
void build_layer(int i, network* net);
void build_conv_layer(int i, network* net);


network* new_network(size_t num_of_layers) {
	network* net = (network*)xcalloc(1, sizeof(network));
	net->n_layers = num_of_layers;
	net->layers = (layer*)xcalloc(num_of_layers, sizeof(layer));
	return net;
}

void build_network(network* net) {
	for (int i = 1; i < net->n_layers; i++) {
		build_layer(i, net);
	}
}

void build_layer(int i, network* net) {
	layer* l = &(net->layers[i]);
	assert(l->type != NONE_LAYER);
	if (i == 0) {
		build_first_layer(net);
		return;
	}
	if (l->type == CONV) {
		build_conv_layer(i, net);
		return;
	}
}

void build_first_layer(network* net) {
	// assume first layer is conv, for now.
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
}

void build_conv_layer(int i, network* net) {
	layer* l = &(net->layers[i]);
	layer* ls = net->layers;
	assert(l->id == i);
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
	size_t c = 0;
	for (size_t j = 0; j < l->in_ids.n; j++) {
		c += ls[j].out_c;
		assert(l->w == ls[j].out_w);
		assert(l->h == ls[j].out_h);
	}
	l->c = c;
	l->n = l->w * l->h * l->c;

	l->in_layers = (layer**)xcalloc(l->in_ids.n, sizeof(layer*));
	for (size_t j = 0; j < l->in_ids.n; j++) {
		l->in_layers[j] = &ls[l->in_ids.a[j]];
	}

	l->out_w = ((l->w + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_h = ((l->h + (l->pad * 2) - l->ksize) / l->stride) + 1;
	l->out_c = l->n_filters;
	l->out_n = l->out_w * l->out_h * l->out_c;

	l->weights.n = l->n_filters * l->ksize * l->ksize * l->c;
	l->weights.a = (float*)xcalloc(l->weights.n, sizeof(float));
	l->biases = (float*)xcalloc(l->n_filters, sizeof(float));
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
	print_layers(n->layers, n->n_layers);
	printf("[END NETWORK]\n");
}

void print_layers(layer* l, size_t num_of_layers) {
	for (size_t i = 0; i < num_of_layers; i++) {
		print_layer(&l[i]);
	}
}

void print_layer(layer* l) {
	printf("\n[LAYER]\n");
	printf("id: %d\n", l->id);
	printf("layer_type: ");
	print_layertype(l->type);
	printf("activation: ");
	print_activation(l->activation);
	printf("batch_size: %zu\n", l->batch_size);
	printf("w, h, c: %zu, %zu, %zu\n", l->w, l->h, l->c);
	printf("n_filters: %zu\n", l->n_filters);
	printf("ksize: %zu\n", l->ksize);
	printf("stride: %zu\n", l->stride);
	printf("pad: %zu\n", l->pad);
	printf("n_inputs: %zu\n", l->n);
	printf("n_outputs: %zu\n", l->out_n);
	printf("n_weights: %zu\n", l->weights.n);
	printf("train: %d\n", l->train);
	printf("batch_norm: %d\n", l->batch_norm);
	printf("in_ids: ");
	print_intarr(&(l->in_ids));
	printf("out_ids: ");
	print_intarr(&(l->out_ids));
	printf("[END LAYER]\n");
}

void print_lrpolicy(LR_POLICY lrp) {
	if (lrp == LR_STEPS) {
		printf("steps\n");
		return;
	}
	printf("none\n");
}

void print_layertype(LAYER_TYPE lt) {
	if (lt == CONV) {
		printf("conv\n");
		return;
	}
	if (lt == YOLO) {
		printf("yolo\n");
		return;
	}
	printf("none\n");
}

void print_activation(ACTIVATION a) {
	if (a == RELU) {
		printf("relu\n");
		return;
	}
	if (a == MISH) {
		printf("mish\n");
		return;
	}
	printf("none\n");
}

//void free_network(network* net) {
//	
//}
