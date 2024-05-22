#include "train.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "utils.h"
#include "image.h"
#include "data.h"
#include "xopencv.h"


void forward_network_train(network* net, sample* samp);
void initialize_weights_kaiming(network* net);
void print_weights(network* net);


void train(network* net) {
	initialize_weights_kaiming(net);
	size_t count[1] = { 0 };

	net->samples = load_samples(net->dp->imgs_dir, count);
	size_t n = count[0];
	net->n_samples = n;
	print_samples(net->samples, n, 1);

	/*for (size_t i = 0; i < n; i++) {
		forward_network_train(net, &net->samples[i]);
	}*/
}

void forward_network_train(network* net, sample* samp) {
	image* img = load_file_to_image(samp->imgpath);
	if (net->w != img->w || net->h != img->h || net->c != img->c) {
		printf("Input image does not match network dimensions.\n");
		printf("img w,h,c = %zu,%zu,%zu\n", img->w, img->h, img->c);
		printf("net w,h,c = %zu,%zu,%zu\n", net->w, net->h, net->c);
		print_location(NARDENET_LOCATION);
		printf("\n\nPress ENTER to exit the program.");
		(void)getchar();
		exit(EXIT_FAILURE);
	}
	for (size_t i = 0; i < net->n_layers; i++) {
		printf("Forwarding layer index %zu...\n", i);
		assert(net->layers[i].forward);
		net->layers[i].forward(&net->layers[i], net);
		printf("Forward done.\n");
	}
	printf("All layers forwarded.\n");
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
