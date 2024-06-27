#include "train.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "utils.h"
#include "image.h"
#include "data.h"
#include "xopencv.h"
#include "costs.h"


void train_classifer(network* net);
void train_detector(network* net);
void forward_network_train(network* net, det_sample* samp);
void initialize_weights_kaiming(network* net);
void print_weights(layer* l);
void print_some_weights(layer* l, size_t n);
void print_top_class_name(float* probs, int n_classes, char** class_names);


#define MAX_DIR_PATH 255
#define MIN_FILENAME_LENGTH 5


void train(network* net) {
	initialize_weights_kaiming(net);
	if (net->type == NET_DETECT) train_detector(net);
	else if (net->type == NET_CLASSIFY) train_classifer(net);
}

void train_classifer(network* net) {
	char train_dir[MAX_DIR_PATH] = { 0 };
	strcpy(train_dir, net->dataset_dir);
	snprintf(train_dir, sizeof(train_dir), "%s%s", train_dir, "train\\");
	load_classifier_dataset(&net->data.clsr, train_dir, net->class_names, net->n_classes);
	layer* prediction_layer = &net->layers[net->n_layers - 1];
	float* truth = prediction_layer->truth;
	float* predictions = prediction_layer->output;
	int n_classes = (int)net->n_classes;
	char** class_names = net->class_names;
	net->max_iterations = 30;  // testing
	layer* layers = net->layers;
	size_t n_layers = net->n_layers;
	for (size_t iter = 0; iter < net->max_iterations; iter++) {
		image* img = get_next_image_classifier_dataset(&net->data.clsr, truth);
		// TODO: Resize img to network dimensions if needed
		if (net->w != img->w || net->h != img->h || net->c != img->c) {
			printf("Input image does not match network dimensions.\n");
			printf("img w,h,c = %zu,%zu,%zu\n", img->w, img->h, img->c);
			printf("net w,h,c = %zu,%zu,%zu\n", net->w, net->h, net->c);
			wait_for_key_then_exit();
		}
		//show_image(img);
		net->input->output = img->data;
		for (size_t i = 0; i < n_layers; i++) {
			layers[i].forward(&layers[i], net);
		}
		for (size_t ii = n_layers; ii; ii--) {
			size_t i = ii - 1;
			layers[i].backprop(&layers[i], net);
		}
		printf("COST = %f\n", prediction_layer->cost);
		for (size_t i = 0; i < n_layers; i++) {
			layers[i].update(&layers[i], net);
		}
		printf("Truth:     ");
		print_top_class_name(truth, n_classes, class_names);
		printf("Predicted: ");
		print_top_class_name(predictions, n_classes, class_names);
		free_image(img);
		net->input->output = NULL;
	}
}

void train_detector(network* net) {
	detector_dataset* data = &net->data.detr;
	data->samples = load_det_samples(net->dataset_dir, &data->n);
	size_t n = data->n;
	for (size_t i = 0; i < n; i++) {
		forward_network_train(net, &data->samples[i]);
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

void print_weights(layer* l) {
	size_t n_weights = l->weights.n;
	float* weights = l->weights.a;
	for (size_t i = 0; i < n_weights; i++) {
		printf("%f\n", weights[i]);
	}
}

void print_top_class_name(float* probs, int n_classes, char** class_names) {
	int c = 0;
	float highscore = 0;
	for (int i = 0; i < n_classes; i++) {
		float p = probs[i];
		if (p > highscore) {
			highscore = p;
			c = i;
		}
	}
	printf("%s (%f)\n", class_names[c], highscore);
}