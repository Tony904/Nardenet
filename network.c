#include "network.h"



network* new_network(int num_of_layers) {  // initialize all struct members to zero
	network* net = (network*)xcalloc(1, sizeof(network));
	net->n_layers = num_of_layers;
	net->layers = (layer*)xcalloc(num_of_layers, sizeof(layer));
	return net;
}

void free_network(network net) {
	free((void*)net.layers);
}

void print_network(network* net) {

}

