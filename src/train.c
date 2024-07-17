#include "train.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "utils.h"
#include "image.h"
#include "data.h"
#include "loss.h"


void train_classifer(network* net);
void train_detector(network* net);
void initialize_weights_kaiming(network* net);
void update_current_learning_rate(network* net, size_t iteration, size_t ease_in);


#define MAX_DIR_PATH 255
#define MIN_FILENAME_LENGTH 5


void train(network* net) {
	net->training = 1;
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
	size_t n_classes = net->n_classes;
	size_t ease_in = net->ease_in;
	size_t batch_size = net->batch_size;
	size_t width = net->w;
	size_t height = net->h;
	size_t channels = net->c;
	float* input = net->input->output;
	//net->max_iterations = 300;  // testing
	layer* layers = net->layers;
	//layer* layer0 = net->layers;
	size_t n_layers = net->n_layers;
	print_network_summary(net, 1);
	for (size_t iter = 0; iter < net->max_iterations; iter++) {
		printf("\nIteration: %zu\n", iter);
		update_current_learning_rate(net, iter, ease_in);
		printf("Learning rate: %f\n", net->current_learning_rate * (float)batch_size);
		classifier_get_next_batch(&net->data.clsr, batch_size, input, width, height, channels, truth, n_classes);
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
	free_classifier_dataset_members(&net->data.clsr);
}

void train_detector(network* net) {
	detector_dataset* data = &net->data.detr;
	data->samples = load_det_samples(net->dataset_dir, &data->n);
	size_t n = data->n;
	for (size_t s = 0; s < n; s++) {
		image* img = load_image(data->samples[s].imgpath);
		if (net->w != img->w || net->h != img->h || net->c != img->c) {
			printf("Input image does not match network dimensions.\n"
				"img w,h,c = %zu,%zu,%zu\nnet w,h,c = %zu,%zu,%zu\n",
				 img->w, img->h, img->c,  net->w, net->h, net->c);
			wait_for_key_then_exit();
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

void update_current_learning_rate(network* net, size_t iteration, size_t ease_in) {
	if (iteration > ease_in) return;
	float power = 4.0F;
	float rate = net->learning_rate * powf((float)iteration / (float)ease_in, power);
	net->current_learning_rate = rate / (float)net->batch_size;
}

