#ifndef CFG_H
#define CFG_H


#include "network.h"
#include "data.h"


#ifdef __cplusplus
extern "C" {
#endif


	network* create_network(data_paths* dp);

	typedef struct cfg_section cfg_section;
	typedef struct cfg_net cfg_net;
	typedef struct cfg_training cfg_training;
	typedef struct cfg_conv cfg_conv;
	typedef struct cfg_classify cfg_classify;

	typedef struct cfg_section {
		char* header;
		void(*set_param) (void* section, char** tokens);
	} cfg_section;

	typedef struct cfg_layer {
		char* header;
		void(*set_param) (void* section, char** tokens);
		int id;
	} cfg_layer;

	typedef struct cfg_net {
		char* header;
		void(*set_param) (void* section, char** tokens);
		size_t width;
		size_t height;
		size_t channels;
		size_t num_classes;
		COST_TYPE cost;
	} cfg_net;

	typedef struct cfg_output {
		char* header;
		void(*set_param) (void* section, char** tokens);
		size_t num_classes;
		COST_TYPE cost;
	} cfg_output;

	typedef struct cfg_training {
		char* header;
		void(*set_param) (void* section, char** tokens);
		size_t batch_size;
		size_t subbatch_size;
		size_t max_iterations;
		float learning_rate;
		LR_POLICY lr_policy;
		floatarr step_percents;
		floatarr step_scaling;
		size_t ease_in;
		float momentum;
		float decay;
		floatarr saturation;
		floatarr exposure;
		floatarr hue;
	} cfg_training;

	typedef struct cfg_conv {
		char* header;
		void(*set_param) (void* section, char** tokens);
		int id;
		int batch_normalize;
		intarr in_ids;
		intarr out_ids;
		size_t n_filters;
		size_t kernel_size;
		size_t stride;
		size_t pad;
		ACTIVATION activation;
		int train;
	} cfg_conv;

	typedef struct cfg_classify {
		char* header;
		void(*set_param) (void* section, char** tokens);
		int id;
		intarr in_ids;
		int train;
		size_t num_classes;
		COST_TYPE cost;
	} cfg_classify;


#ifdef __cplusplus
}
#endif
#endif