#include "train.h"

#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "image.h"


void initialize_weights_kaiming(network* net);
void print_weights(network* net);


void train(network* net) {
	initialize_weights_kaiming(net);

}

void forward_network_train(network* net, image* img) {
	if (net->w != img->w || net->h != img->h || net->c != img->h) {
		printf("Input image does not match network dimensions.\n");
		print_location(NARDENET_LOCATION);
		exit(EXIT_FAILURE);
	}
	net->input = img;
	for (size_t i = 0; net->n_layers; i++) {
		net->layers[i].forward(&net->layers[i], net);
	}
	// float* prediction = network_get_prediction;
}

// Initialize weights using Kaiming Initialization
void initialize_weights_kaiming(network* net) {
	printf("\nInitializing weights...");
	srand(7777777);
	layer* layers = net->layers;
	size_t N = net->n_layers;
	size_t n;
	layer* l;
	double stddev;
	for (size_t i = 0; i < N; i++) {
		l = &layers[i];
		n = l->weights.n;
		stddev = sqrt(2.0 / n);
		for (size_t j = 0; j < n; j++) {
			l->weights.a[j] = (float)(stddev * randn(0.0, 1));
		}
	}
	printf(" done\n");
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
			if ((j + 1 ) % 10 == 0) printf("\n");
		}
	}
}
