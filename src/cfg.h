#ifndef CFG_H
#define CFG_H


#include "network.h"
#include "list.h"
#include "utils.h"


#ifdef __cplusplus
extern "C" {
#endif

	typedef struct cfg cfg;
	typedef struct cfg_section cfg_section;

	network* create_network_from_cfg(char* cfgfile);

	typedef struct cfg {
		// [data]
		char* dataset_dir;
		char* classes_file;
		char* weights_file;
		char* backup_dir;
		// [net]
		size_t width;
		size_t height;
		size_t channels;
		size_t n_classes;
		// [training]
		size_t batch_size;
		size_t subbatch_size;
		size_t max_iterations;
		float learning_rate;
		LR_POLICY lr_policy;
		floatarr step_percents;
		floatarr step_scaling;
		size_t ease_in;
		float momentum;
		REGULARIZATION regularization;
		float decay;
		floatarr saturation;
		floatarr exposure;
		floatarr hue;
		// layers
		list* layers;
	} cfg;

	typedef struct cfg_layer {
		LAYER_TYPE type;
		int id;
		int train;
		intarr in_ids;
		intarr out_ids;
		int batch_normalize;
		size_t n_filters;
		size_t kernel_size;
		size_t stride;
		size_t pad;
		ACTIVATION activation;
		// [classify] or [detect]
		size_t n_classes;
		LOSS_TYPE loss_type;
	} cfg_layer;

#ifdef __cplusplus
}
#endif
#endif