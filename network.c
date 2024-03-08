#include "network.h"



network make_network(int num_of_layers) {
	network net = { 0 };  // initialize all struct members to zero
	net.n_layers = num_of_layers;
	net.layers = (layer*)xcalloc(net.n_layers, sizeof(layer));
	return net;
}

void free_network(network net) {
	free((void*)net.layers);
}

