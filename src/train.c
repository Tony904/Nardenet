#include "train.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "utils.h"
#include "image.h"
#include "data.h"
#include "xopencv.h"


void forward_network_train(network* net, det_sample* samp);
void initialize_weights_kaiming(network* net);
void print_weights(network* net);


#define MAX_DIR_PATH 255
#define MIN_FILENAME_LENGTH 5


void train(network* net) {
	initialize_weights_kaiming(net);
	if (net->type == NET_DETECT) train_detector(net);
	else if (net->type == NET_CLASSIFY) train_classifer(net);

	/*for (size_t i = 0; i < n; i++) {
		forward_network_train(net, &net->det_samples[i]);
	}*/
}

void train_classifer(network* net) {
	char* train_dir[MAX_DIR_PATH] = { 0 };
	strcpy(train_dir, net->dataset_dir);
	snprintf(train_dir, sizeof(train_dir), "%s%s", train_dir, "train\\");
	net->data.sets = load_class_sets(train_dir, net->class_names, &net->data.n);

	net->input = load_img_from_classifier_dataset(lbl, i);
}

void train_detector(network* net) {
	net->data.samples = load_det_samples(net->dataset_dir, &net->data.n);
	for (size_t i = 0; i < n; i++) {
		forward_network_train(net, &net->data.samples[i]);
	}
}

void forward_network_train(network* net, det_sample* samp) {
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
