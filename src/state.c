#include "state.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "locmacro.h"
#include "utils.h"
#include "xallocs.h"
#include "blas.h"


size_t write_floats(float* data, size_t n, FILE* file, char* data_name, int layer_id);
size_t read_floats(float* dst, size_t n, FILE* file, char* data_name, int layer_id);
void close_file_then_exit(FILE* file);

#define SIGNATURE 8312024
#define STATE_VERSION 1
#define NARDENET_VERSION 1
#define HEADER_SIZE 20


void save_state(network* net) {
	char filename[_MAX_PATH] = { 0 };
	strcpy(filename, net->weights_file);
	int ext_i = get_filename_ext_index(filename);
	char suffix[_MAX_PATH] = { 0 };
	snprintf(suffix, sizeof(suffix), "-%zu", net->iteration);
	insert_chars(filename, sizeof(filename), ext_i, suffix);
	printf("Saving network state to %s\n", filename);
	FILE* file = get_filestream(filename, "wb");
	uint64_t h[HEADER_SIZE] = { 0 };
	h[0] = SIGNATURE;
	h[1] = STATE_VERSION;
	h[2] = NARDENET_VERSION;
	uint64_t total_vals = 0;
	h[3] = total_vals;
	h[4] = (uint64_t)net->n_layers;
	h[5] = (uint64_t)net->iteration;
	h[6] = (uint64_t)net->w;
	h[7] = (uint64_t)net->h;
	h[8] = (uint64_t)net->c;
	h[9] = (uint64_t)net->n_classes;
	size_t written = fwrite(h, sizeof(uint64_t), HEADER_SIZE, file);
	if (written != HEADER_SIZE) {
		printf("Error writing state header to file.\n");
		close_file_then_exit(file);
	}
	size_t n_layers = net->n_layers;
	layer* layers = net->layers;
	for (size_t i = 0; i < n_layers; i++) {
		layer* l = &layers[i];
		if (l->type == LAYER_MAXPOOL || l->type == LAYER_RESIDUAL || l->type == LAYER_DETECT) {
			continue;
		}
		int id = l->id;
		total_vals += write_floats(l->weights, l->n_weights, file, "weights", id);
		total_vals += write_floats(l->biases, l->n_filters, file, "biases", id);
		total_vals += write_floats(l->weight_velocities, l->n_weights, file, "weight_velocities", id);
		total_vals += write_floats(l->bias_velocities, l->n_filters, file, "bias_velocities", id);
		if (l->batchnorm) {
			total_vals += write_floats(l->gammas, l->n_filters, file, "gammas", id);
			total_vals += write_floats(l->rolling_means, l->n_filters, file, "rolling_means", id);
			total_vals += write_floats(l->rolling_variances, l->n_filters, file, "rolling_variances", id);
			total_vals += write_floats(l->gamma_velocities, l->n_filters, file, "gamma_velocities", id);
		}
	}
	if (fseek(file, sizeof(uint64_t) * 3, SEEK_SET)) {
		printf("Error moving file pointer while writing state to file.");
		close_file_then_exit(file);
	}
	if (total_vals > 0) {
		written = fwrite(&total_vals, sizeof(uint64_t), 1, file);
		if (written != 1) {
			printf("Error writing total_vals to file.");
			close_file_then_exit(file);
		}
	}
	close_filestream(file);
	printf("Save complete.\n");
}

void load_state(network* net) {
	printf("Loading network state from %s\n", net->weights_file);
	FILE* file = get_filestream(net->weights_file, "rb");
	uint64_t h[HEADER_SIZE] = { 0 };
	size_t read = fread(h, sizeof(uint64_t), HEADER_SIZE, file);
	if (ferror(file)) {
		printf("Error reading header from file %s", net->weights_file);
		close_file_then_exit(file);
	}
	if (read != HEADER_SIZE) {
		printf("Error reading state header from file.\n");
		close_file_then_exit(file);
	}
	if (h[0] != SIGNATURE) {
		printf("Invalid file signature in %s\n", net->weights_file);
		close_file_then_exit(file);
	}
	if (h[1] != (uint64_t)STATE_VERSION) {
		printf("Invalid state version of file %s\n", net->weights_file);
		printf("State version installed: %zu, read: %zu\n", (size_t)STATE_VERSION, (size_t)h[1]);
		close_file_then_exit(file);
	}
	if (h[2] != (uint64_t)NARDENET_VERSION) {
		printf("Invalid Nardenet version of file %s\n", net->weights_file);
		printf("Nardenet version installed: %zu, read: %zu\n", (size_t)NARDENET_VERSION, (size_t)h[2]);
		close_file_then_exit(file);
	}
	if ((size_t)h[4] != net->n_layers) {
		printf("# of layers in cfg file does not match # in %s\n", net->weights_file);
		close_file_then_exit(file);
	}
	net->iteration = (size_t)h[5];
	if (h[6] != net->w) {
		printf("Network width in cfg file does not match the width in %s\n", net->weights_file);
		close_file_then_exit(file);
	}
	if (h[7] != net->h) {
		printf("Network height in cfg file does not match the height in %s\n", net->weights_file);
		close_file_then_exit(file);
	}
	if (h[8] != net->c) {
		printf("Network channels in cfg file does not match the channels in %s\n", net->weights_file);
		close_file_then_exit(file);
	}
	if (h[9] != net->n_classes) {
		printf("# of classes in cfg file does not match the # in %s\n", net->weights_file);
		close_file_then_exit(file);
	}
	uint64_t total_vals = 0;
	size_t n_layers = net->n_layers;
	layer* layers = net->layers;
	for (size_t i = 0; i < n_layers; i++) {
		layer* l = &layers[i];
		if (l->type == LAYER_MAXPOOL || l->type == LAYER_RESIDUAL || l->type == LAYER_DETECT) {
			continue;
		}
		int id = l->id;
		total_vals += read_floats(l->weights, l->n_weights, file, "weights", id);
		total_vals += read_floats(l->biases, l->n_filters, file, "biases", id);
		total_vals += read_floats(l->weight_velocities, l->n_weights, file, "weight_velocities", id);
		total_vals += read_floats(l->bias_velocities, l->n_filters, file, "bias_velocities", id);
		if (l->batchnorm) {
			total_vals += read_floats(l->gammas, l->n_filters, file, "gammas", id);
			total_vals += read_floats(l->rolling_means, l->n_filters, file, "rolling_means", id);
			total_vals += read_floats(l->rolling_variances, l->n_filters, file, "rolling_variances", id);
			total_vals += read_floats(l->gamma_velocities, l->n_filters, file, "gamma_velocities", id);
		}
	}
	if (total_vals != h[3]) {
		printf("Total values in state file does not match actual # of values read.\n");
		close_file_then_exit(file);
	}
	close_filestream(file);
}

/* n = # of floats to write to file.
Returns # of floats written.*/
size_t write_floats(float* data, size_t n, FILE* file, char* data_name, int layer_id) {
	if (!n) return 0;
	if (!data) return 0;
	size_t written = fwrite(data, sizeof(float), n, file);
	if (written != n) {
		printf("Error saving layer %d %s.\n", layer_id, data_name);
		close_file_then_exit(file);
	}
	return written;
}

/* n = # of floats to read from file and store in dst.
Returns # of floats read.*/
size_t read_floats(float* dst, size_t n, FILE* file, char* data_name, int layer_id) {
	if (!n) return 0;
	if (!dst) return 0;
	size_t floats_read = fread((void*)dst, sizeof(float), n, file);
	if (ferror(file) || floats_read != n) {
		printf("Error loading layer %d %s.\n", layer_id, data_name);
		close_file_then_exit(file);
	}
	return floats_read;
}

void close_file_then_exit(FILE* file) {
	close_filestream(file);
	wait_for_key_then_exit();
}

/*** TESTING ***/

void test_save_state(void) {
	network net = { 0 };
	net.n_layers = 2;
	net.w = 5;
	net.h = 5;
	net.c = 3;
	net.iteration = 123;
	net.n_classes = 2;
	net.layers = (layer*)xcalloc(net.n_layers, sizeof(layer));
	layer* l1 = &net.layers[0];
	l1->n_weights = 10;
	l1->weights = (float*)xcalloc(l1->n_weights, sizeof(float));
	fill_array_increment(l1->weights, l1->n_weights, 0.0F, 1.0F);
	layer* l2 = &net.layers[1];
	l2->n_weights = 4;
	l2->weights = (float*)xcalloc(l2->n_weights, sizeof(float));
	fill_array_increment(l2->weights, l2->n_weights, 7.0F, 2.0F);
	l1->n_filters = 0;
	net.weights_file = "C:\\Users\\TNard\\OneDrive\\Desktop\\dev\\Nardenet-main\\data\\test_weights.txt";
	save_state(&net);
	zero_array(l1->weights, l1->n_weights);
	net.weights_file = "C:\\Users\\TNard\\OneDrive\\Desktop\\dev\\Nardenet-main\\data\\test_weights-123.txt";
	load_state(&net);
	print_network(&net);
	print_all_network_weights(&net);
}

void test_load_state(void) {
	network net = { 0 };
	net.n_layers = 2;
	net.w = 5;
	net.h = 5;
	net.c = 3;
	net.iteration = 123;
	net.n_classes = 2;
	net.layers = (layer*)xcalloc(net.n_layers, sizeof(layer));
	layer* l1 = &net.layers[0];
	l1->n_weights = 10;
	l1->weights = (float*)xcalloc(l1->n_weights, sizeof(float));
	fill_array_increment(l1->weights, l1->n_weights, 0.0F, 1.0F);
	layer* l2 = &net.layers[1];
	l2->n_weights = 4;
	l2->weights = (float*)xcalloc(l2->n_weights, sizeof(float));
	fill_array_increment(l2->weights, l2->n_weights, 7.0F, 2.0F);
	l1->n_filters = 0;
	net.weights_file = "C:\\Users\\TNard\\OneDrive\\Desktop\\dev\\Nardenet-main\\data\\test_weights.txt";
	save_state(&net);
	zero_array(l1->weights, l1->n_weights);
	net.weights_file = "C:\\Users\\TNard\\OneDrive\\Desktop\\dev\\Nardenet-main\\data\\test_weights-123.txt";
	load_state(&net);
	print_network(&net);
	print_all_network_weights(&net);
}