#ifndef CONFIG_H
#define CONFIG_H

#include "nardenet.h"
#include "list.h"


#ifdef __cplusplus
extern "C" {
#endif


	void load_cfg(char* filename, network* net);

	typedef struct cfg_section cfg_section;
	typedef struct cfg_input cfg_input;
	typedef struct cfg_training cfg_training;
	typedef struct cfg_conv_layer cfg_conv_layer;


	typedef struct cfg_section {
		char* header;
		void(*set_param) (void* section, char** tokens);
	};

	typedef struct cfg_input {
		char* header;
		void(*set_param) (struct cfg_input, char** tokens);
		int width;
		int height;
		int channels;
	} cfg_input;

	typedef struct cfg_training {
		char* header;
		void(*set_param) (struct cfg_training, char** tokens);
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

	typedef struct cfg_conv {
		char* header;
		void(*set_param) (struct cfg_conv, char** tokens);
		int id;
		int batch_normalize;
		int n_filters;
		int kernel_size;
		int stride;
		int pad;
		ACTIVATION activation;
	} cfg_conv;

	typedef struct cfg_yolo {
		char* header;
		void(*set_param) (struct cfg_yolo, char** tokens);
		int id;
		int n_filters;
		int kernel_size;
		int stride;
		int pad;
		int* anchors;
	} cfg_yolo;


#ifdef def __cplusplus
}
#endif
#endif