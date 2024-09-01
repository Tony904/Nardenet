#include "state.h"
#include <stdio.h>
#include <stdint.h>
#include "locmacro.h"
#include "utils.h"



int write_binary_float(float* data, size_t n, FILE* file);
int read_binary_float(float* dst, size_t n, FILE* file);
void close_file_then_exit(FILE* file);

#define SIGNATURE 8312024
#define STATE_VERSION 1
#define NARDENET_VERSION 1
#define HEADER_SIZE 20


void write_state(network* net) {
	FILE* file = get_filestream(net->weights_file, "wb");
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
		if (!write_binary_float(l->weights.a, l->weights.n, file)) {
			printf("Error writing weights to file. layer: %d\n", l->id);
			close_file_then_exit(file);
		}
		total_vals += l->weights.n;
		if (!write_binary_float(l->biases, l->n_filters, file)) {
			printf("Error writing biases to file. layer: %d\n", l->id);
			close_file_then_exit(file);
		}
		total_vals += l->n_filters;
		if (!write_binary_float(l->weights_velocity, l->weights.n, file)) {
			printf("Error writing weights_velocity to file. layer: %d\n", l->id);
			close_file_then_exit(file);
		}
		total_vals += l->weights.n;
		if (!write_binary_float(l->biases_velocity, l->n_filters, file)) {
			printf("Error writing biases_velocity to file. layer: %d\n", l->id);
			close_file_then_exit(file);
		}
		total_vals += l->n_filters;
		if (l->batch_norm) {
			if (!write_binary_float(l->gammas, l->n_filters, file)) {
				printf("Error writing gammas to file. layer: %d\n", l->id);
				close_file_then_exit(file);
			}
			total_vals += l->n_filters;
			if (!write_binary_float(l->rolling_means, l->n_filters, file)) {
				printf("Error writing rolling means to file. layer: %d\n", l->id);
				close_file_then_exit(file);
			}
			total_vals += l->n_filters;
			if (!write_binary_float(l->rolling_variances, l->n_filters, file)) {
				printf("Error writing rolling variances to file. layer: %d\n", l->id);
				close_file_then_exit(file);
			}
			total_vals += l->n_filters;
			if (!write_binary_float(l->gammas_velocity, l->n_filters, file)) {
				printf("Error writing gammas_velocity to file. layer: %d\n", l->id);
				close_file_then_exit(file);
			}
			total_vals += l->n_filters;
		}
	}
	if (!fseek(file, sizeof(uint64_t) * 3, SEEK_SET)) {
		printf("Error moving file pointer while writing state to file.");
		close_file_then_exit(file);
	}
	if (total_vals > 0) {
		written = fwrite(total_vals, sizeof(uint64_t), 1, file);
		if (written != 1) {
			printf("Error writing total_vals to file.\n");
			close_file_then_exit(file);
		}
	}
	close_filestream(file);
}

void load_state(network* net) {
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
		if (!read_binary_float(l->weights.a, l->weights.n, file)) {
			printf("Error reading weights from file. layer: %d\n", l->id);
			close_file_then_exit(file);
		}
		total_vals += l->weights.n;
		if (!read_binary_float(l->biases, l->n_filters, file)) {
			printf("Error reading biases from file. layer: %d\n", l->id);
			close_file_then_exit(file);
		}
		total_vals += l->n_filters;
		if (!read_binary_float(l->weights_velocity, l->weights.n, file)) {
			printf("Error reading weights_velocity from file. layer: %d\n", l->id);
			close_file_then_exit(file);
		}
		total_vals += l->weights.n;
		if (!read_binary_float(l->biases_velocity, l->n_filters, file)) {
			printf("Error reading biases_velocity from file. layer: %d\n", l->id);
			close_file_then_exit(file);
		}
		total_vals += l->n_filters;
		if (l->batch_norm) {
			if (!read_binary_float(l->gammas, l->n_filters, file)) {
				printf("Error reading gammas from file. layer: %d\n", l->id);
				close_file_then_exit(file);
			}
			total_vals += l->n_filters;
			if (!read_binary_float(l->rolling_means, l->n_filters, file)) {
				printf("Error reading rolling means from file. layer: %d\n", l->id);
				close_file_then_exit(file);
			}
			total_vals += l->n_filters;
			if (!read_binary_float(l->rolling_variances, l->n_filters, file)) {
				printf("Error reading rolling variances from file. layer: %d\n", l->id);
				close_file_then_exit(file);
			}
			total_vals += l->n_filters;
			if (!read_binary_float(l->gammas_velocity, l->n_filters, file)) {
				printf("Error reading gammas_velocity from file. layer: %d\n", l->id);
				close_file_then_exit(file);
			}
			total_vals += l->n_filters;
		}
	}
	if (total_vals != h[3]) {
		printf("Total values in state file does not match actual # of values read.\n");
		close_file_then_exit(file);
	}
	close_filestream(file);
}

/* n = # of floats to write to file.
Returns 1 if successful, otherwise 0. */
int write_binary_float(float* data, size_t n, FILE* file) {
	size_t written = fwrite(data, sizeof(float), n, file);
	return (written == n);
}

/* n = # of floats to read from file and store in dst.
Returns 1 if successful, otherwise 0.*/
int read_binary_float(float* dst, size_t n, FILE* file) {
	if (!n) return 0;
	size_t floats_read = fread((void*)dst, sizeof(float), n, file);
	return (!ferror(file) && floats_read == n);
}

void close_file_then_exit(FILE* file) {
	close_filestream(file);
	wait_for_key_then_exit();
}