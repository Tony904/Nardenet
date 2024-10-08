#include "train.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "utils.h"
#include "image.h"
#include "data.h"
#include "loss.h"
#include "state.h"
#include "xallocs.h"


void train_classifer(network* net);
void train_detector(network* net);
void initialize_weights_kaiming(network* net);
void update_current_learning_rate(network* net, size_t iteration);


#define MAX_DIR_PATH _MAX_PATH - 5


void train(network* net) {
	net->training = 1;
	if (net->weights_file) load_state(net);
	else {
		printf("No weights file specified. Using random weights.\n");
		initialize_weights_kaiming(net);
		net->weights_file = (char*)xcalloc(_MAX_PATH, sizeof(char));
		char buf[_MAX_PATH] = { 0 };
		get_filename_from_path(buf, _MAX_PATH, net->cfg_file, 1);
		snprintf(net->weights_file, _MAX_PATH, "%s%s%s", net->save_dir, buf, ".weights");
	}
	if (net->type == NET_DETECT) train_detector(net);
	else if (net->type == NET_CLASSIFY) train_classifer(net);
}

void train_classifer(network* net) {
	char train_dir[MAX_DIR_PATH] = { 0 };
	strcpy(train_dir, net->dataset_dir);
	snprintf(train_dir, sizeof(train_dir), "%s%s", train_dir, "train\\");
	load_classifier_dataset(&net->data.clsr, train_dir, net->class_names, net->n_classes);
	size_t batch_size = net->batch_size;
	size_t max_iterations = net->max_iterations;
	size_t save_frequency = net->save_frequency;
	layer* layers = net->layers;
	size_t n_layers = net->n_layers;
	print_network_summary(net, 1);
	print_network_structure(net);
	for (size_t iter = 0; iter < max_iterations; iter++) {
		printf("\nIteration: %zu\n", iter + 1);
		update_current_learning_rate(net, iter);
		printf("Learning rate: %f\n", net->current_learning_rate * (float)batch_size);
		classifier_get_next_batch(net);
		for (size_t i = 0; i < n_layers; i++) {
			layers[i].forward(&layers[i], net);
		}
		if (net->regularization != REG_NONE) {
			net->reg_loss(net);
			printf("Regularization loss: %f\n", net->loss);
		}
		for (size_t ii = n_layers; ii; ii--) {
			size_t i = ii - 1;
			layers[i].backward(&layers[i], net);
		}
		for (size_t i = 0; i < n_layers; i++) {
			layers[i].update(&layers[i], net);
		}
		if (save_frequency) {
			if ((iter + 1) % save_frequency == 0) {
				net->iteration = iter + 1;
				save_state(net);
			}
		}
	}
	free_classifier_dataset_members(&net->data.clsr);
}

void train_detector(network* net) {
	char train_dir[MAX_DIR_PATH] = { 0 };
	memcpy(train_dir, net->dataset_dir, strlen(net->dataset_dir));
	memcpy(&train_dir[strlen(train_dir)], "\\train\\", 7);
	load_detector_dataset(&net->data.detr, train_dir);
	size_t batch_size = net->batch_size;
	layer* layers = net->layers;
	size_t n_layers = net->n_layers;
	print_network_summary(net, 1);
	for (size_t iter = 0; iter < net->max_iterations; iter++) {
		printf("\nIteration: %zu\n", iter);
		update_current_learning_rate(net, iter);
		printf("Learning rate: %f\n", net->current_learning_rate * (float)batch_size);
		detector_get_next_batch(net);
		for (size_t i = 0; i < n_layers; i++) {
			layers[i].forward(&layers[i], net);
		}
		if (net->regularization != REG_NONE) {
			net->reg_loss(net);
			printf("Regularization loss: %f\n", net->loss);
		}
		for (size_t ii = n_layers; ii; ii--) {
			size_t i = ii - 1;
			layers[i].backward(&layers[i], net);
		}
		for (size_t i = 0; i < n_layers; i++) {
			layers[i].update(&layers[i], net);
		}
	}
}

/*Initialize weights using Kaiming Initialization*/
void initialize_weights_kaiming(network* net) {
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
}

void update_current_learning_rate(network* net, size_t iteration) {
	size_t ease_in = net->ease_in;
	float rate = net->learning_rate;
	float max_iterations = (float)net->max_iterations;
	for (size_t n = 0; n < net->step_percents.n; n++) {
		if (iteration >= net->step_percents.a[n] * max_iterations) rate *= net->step_scaling.a[n];
		else break;
	}
	if (iteration < ease_in) rate *= powf((float)iteration / (float)ease_in, 4.0F);
	net->current_learning_rate = rate / (float)net->batch_size;
}

