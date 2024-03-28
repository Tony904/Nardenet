#ifndef CFG_H
#define CFG_H


#include "network.h"


#ifdef __cplusplus
extern "C" {
#endif


	network* create_network_from_cfg(char* filename);

	typedef struct cfg_section cfg_section;
	typedef struct cfg_input cfg_input;
	typedef struct cfg_training cfg_training;
	typedef struct cfg_conv_layer cfg_conv_layer;

	typedef struct cfg_section {
		char* header;
		void(*set_param) (void* section, char** tokens);
	} cfg_section;

	typedef struct cfg_layer {
		char* header;
		void(*set_param) (void* section, char** tokens);
		int id;
	} cfg_layer;

	typedef struct cfg_input {
		char* header;
		void(*set_param) (void* section, char** tokens);
		size_t width;
		size_t height;
		size_t channels;
	} cfg_input;

	typedef struct cfg_training {
		char* header;
		void(*set_param) (void* section, char** tokens);
		size_t batch_size;
		size_t subbatch_size;
		size_t max_iterations;
		float learning_rate;
		LR_POLICY lr_policy;
		floatarr* step_percents;
		floatarr* step_scaling;
		size_t ease_in;
		float momentum;
		float decay;
		floatarr* saturation;
		floatarr* exposure;
		floatarr* hue;
	} cfg_training;

	typedef struct cfg_conv {
		char* header;
		void(*set_param) (void* section, char** tokens);
		int id;
		int batch_normalize;
		size_t n_filters;
		size_t kernel_size;
		size_t stride;
		size_t pad;
		ACTIVATION activation;
	} cfg_conv;

	typedef struct cfg_yolo {
		char* header;
		void(*set_param) (void* section, char** tokens);
		int id;
		size_t n_filters;
		size_t kernel_size;
		size_t stride;
		size_t pad;
		size_t* anchors;
	} cfg_yolo;


#ifdef __cplusplus
}
#endif
#endif