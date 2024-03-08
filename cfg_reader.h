#ifndef CFG_READER_H
#define CFG_READER_H

#include "nardenet.h"

#ifdef __cplusplus
extern "C" {
#endif

void load_cfg(char* filename, network net);

typedef struct cfg_input {
	int width;
	int height;
	int channels;
} cfg_input;

typedef struct cfg_training {
	int batch_size;
	int subbatch_size;
	int max_iterations;
	float learning_rate;
	LR_POLICY lr_policy;
	float* step_percents;
	float* step_scaling;
	int ease_in;
	float momentum;
	float decay;
	float saturation[2];
	float exposure[2];
	float hue[2];
} cfg_training;

typedef struct cfg_conv_layer {
	int id;
	int batch_normalize;
	int n_filters;
	int kernel_size;
	int stride;
	int pad;
	ACTIVATION activation;
} cfg_conv_layer;

#ifdef __cplusplus
}
#endif
#endif