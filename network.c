#include "network.h"
#include "xallocs.h"
#include "utils.h"
#include <stdlib.h>


void print_layers(layer* s, size_t num_of_layers);
void print_layer(layer* s);


network* new_network(size_t num_of_layers) {  // initialize all struct members to zero
	network* net = (network*)xcalloc(1, sizeof(network));
	net->n_layers = num_of_layers;
	net->layers = (layer*)xcalloc(num_of_layers, sizeof(layer));
	return net;
}

void free_network(network* net) {
	xfree(net->layers);
	xfree(net->output);
	xfree(net->layers);
	xfree(net);
}

void print_network(network* n) {
	printf("[NETWORK]\n");
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
	printf("[END NETWORK]");
}

void print_layers(layer* s, size_t num_of_layers) {
	for (size_t i = 0; i < num_of_layers; i++) {
		print_layer(&s[i]);
	}
}

void print_layer(layer* s) {
	printf("\n[LAYER]\n");
	printf("id: %d\n", s->id);
	printf("layer_type: ");
	print_layertype(s->type);
	printf("activation: ");
	print_activation(s->activation);
	printf("batch_size: %zu\n", s->batch_size);
	printf("w, h, c: %zu, %zu, %zu\n", s->w, s->h, s->c);
	printf("n_filters: %zu\n", s->n_filters);
	printf("k_size: %zu\n", s->k_size);
	printf("stride: %zu\n", s->stride);
	printf("padding: %zu\n", s->padding);
	printf("n_inputs: %zu\n", s->n_inputs);
	printf("n_outputs: %zu\n", s->n_outputs);
	printf("n_weights: %zu\n", s->n_weights);
	printf("train: %d\n", s->train);
	printf("batch_norm: %d\n", s->batch_norm);
	printf("[END LAYER]\n");
}