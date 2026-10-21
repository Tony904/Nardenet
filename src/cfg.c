#include "cfg.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "xallocs.h"
#include "utils.h"


#define hDATA "[data]"
#define hNET "[net]"
#define hTRAINING "[training]"
#define hCONV "[conv]"
#define hDENSE "[dense]"
#define hMAXPOOL "[maxpool]"
#define hRESIDUAL "[residual]"
#define hCLASSIFY "[classify]"
#define hDETECT "[detect]"
#define hAVGPOOL_GLOBAL "[avgpool global]"
#define hUPSAMPLE "[upsample]"
#define hROUTE "[route]"

#define LINESIZE 512
#define TOKENSSIZE 64


void load_cfg(char* filename, network_cfg* c);
void copy_to_network_cfg(network_cfg* c, char** tokens, char* header);
void copy_to_layer_cfg(layer_cfg* l, char** tokens, network_cfg* c);
void copy_cfg_to_network(network_cfg* net_cfg, network* net);
void copy_data_augment_range(float* dst, floatarr src);
void load_class_names(char* filename, network* net, size_t n_classes);
LR_POLICY str2lr_policy(char* str);
ACTIVATION str2activation(char* str);
LOSS_TYPE str2loss(char* str);
REGULARIZATION str2regularization(char* str);
floatarr tokens2floatarr(char** tokens);
intarr tokens2intarr(char** tokens);
void parse_chunk_args(layer_cfg* l, char** tokens);
int is_header(char* str, char* header, int* is_layer);
LAYER_TYPE header2layertype(char* header);
void print_network_cfg(network_cfg* c);
void print_layer_cfg(layer_cfg* l);


network* create_network_from_cfg(char* cfgfile) {
	network_cfg c = { 0 };
	printf("Loading cfg file: %s\n", cfgfile);
	load_cfg(cfgfile, &c);
	if (!c.layers) {
		printf("No layers in cfg.\n");
		wait_for_key_then_exit();
	}
#pragma warning (suppress:6011)  // Visual Studio is too stupid to recognize c.layers can't be null here
	network* net = new_network(c.layers->length);
	copy_cfg_to_network(&c, net);
	net->cfg_file = cfgfile;
	xfree(&c.classes_file);
	xfree(&c.saturation.a);
	xfree(&c.exposure.a);
	xfree(&c.hue.a);
	free_list(c.layers, 1);
	build_network(net);
	return net;
}

void load_cfg(char* filename, network_cfg* c) {
	FILE* file = get_filestream(filename, "r");
	char line[LINESIZE] = { 0 };
	char* tokens[TOKENSSIZE] = { 0 };
	char header[50] = { 0 };
	layer_cfg* l = 0;
	list* layers = (list*)xcalloc(1, sizeof(list));
	int is_layer = 0;
	int layer_id = 0;
	while (read_line_to_buff(file, line, LINESIZE)) {
		clean_string(line);
		split_string_to_buff(line, "=,", tokens, TOKENSSIZE);
		if (!tokens[0]) continue;
		if (is_header(line, header, &is_layer)) {
			if (!is_layer) continue;
			l = (layer_cfg*)xcalloc(1, sizeof(layer_cfg));
			l->id = layer_id;
			layer_id++;
			l->train = 1;  // default
			list_append(layers, l);
			l->type = header2layertype(header);
		}
		else if (!is_layer) {
			copy_to_network_cfg(c, tokens, header);
		}
		else {
			copy_to_layer_cfg(l, tokens, c);
		}
	}
	c->layers = layers;
	close_filestream(file);
}

void copy_to_network_cfg(network_cfg* c, char** tokens, char* header) {
	size_t t = tokens_length(tokens);
	if (t < 2) {
		printf("Error. Tokens must have a minimum length of 2, has %zu.\n", t);
		printf("Tokens:\n");
		print_str_array(tokens, t);
		wait_for_key_then_exit();
	}
	char* k = tokens[0];
	if (strcmp(header, hDATA) == 0) {
		if (strcmp(k, "dataset_dir") == 0) {
			c->dataset_dir = (char*)xcalloc(_MAX_PATH, sizeof(char));
			strcpy(c->dataset_dir, tokens[1]);
			fix_dir_str(c->dataset_dir, _MAX_PATH);
		}
		else if (strcmp(k, "classes_file") == 0) {
			c->classes_file = (char*)xcalloc(strlen(tokens[1]) + 1, sizeof(char));
			strcpy(c->classes_file, tokens[1]);
		}
		else if (strcmp(k, "weights_file") == 0) {
			c->weights_file = (char*)xcalloc(strlen(tokens[1]) + 1, sizeof(char));
			strcpy(c->weights_file, tokens[1]);
		}
		else if (strcmp(k, "save_dir") == 0) {
			c->save_dir = (char*)xcalloc(_MAX_PATH, sizeof(char));
			strcpy(c->save_dir, tokens[1]);
			fix_dir_str(c->save_dir, _MAX_PATH);
		}
	}
	else if (strcmp(header, hNET) == 0) {
		if (strcmp(k, "width") == 0) {
			c->width = str2sizet(tokens[1]);
		}
		else if (strcmp(k, "height") == 0) {
			c->height = str2sizet(tokens[1]);
		}
		else if (strcmp(k, "channels") == 0) {
			c->channels = str2sizet(tokens[1]);
		}
		else if (strcmp(k, "num_classes") == 0) {
			c->n_classes = str2sizet(tokens[1]);
		}
		else if (strcmp(k, "use_gpu") == 0) {
			c->use_gpu = str2int(tokens[1]);
		}
	}
	else if (strcmp(header, hTRAINING) == 0) {
		if (strcmp(k, "batch_size") == 0) {
			c->batch_size = str2sizet(tokens[1]);
		}
		else if (strcmp(k, "subbatch_size") == 0) {
			c->subbatch_size = str2sizet(tokens[1]);
		}
		else if (strcmp(k, "max_iterations") == 0) {
			c->max_iterations = str2sizet(tokens[1]);
		}
		else if (strcmp(k, "save_frequency") == 0) {
			c->save_frequency = str2sizet(tokens[1]);
		}
		else if (strcmp(k, "learning_rate") == 0) {
			c->learning_rate = str2float(tokens[1]);
		}
		else if (strcmp(k, "lr_policy") == 0) {
			c->lr_policy = str2lr_policy(tokens[1]);
		}
		else if (strcmp(k, "step_percents") == 0) {
			c->step_percents = tokens2floatarr(&tokens[1]);
		}
		else if (strcmp(k, "step_scaling") == 0) {
			c->step_scaling = tokens2floatarr(&tokens[1]);
		}
		else if (strcmp(k, "coswr_frequency") == 0) {
			c->coswr_frequency = str2sizet(tokens[1]);
			if (c->coswr_multi == 0.0F) c->coswr_multi = 1.0F;
		}
		else if (strcmp(k, "coswr_multi") == 0) {
			c->coswr_multi = str2float(tokens[1]);
		}
		else if (strcmp(k, "exp_decay") == 0) {
			c->exp_decay = str2float(tokens[1]);
		}
		else if (strcmp(k, "poly_pow") == 0) {
			c->poly_pow = str2float(tokens[1]);
		}
		else if (strcmp(k, "ease_in") == 0) {
			c->ease_in = str2sizet(tokens[1]);
		}
		else if (strcmp(k, "momentum") == 0) {
			c->momentum = str2float(tokens[1]);
		}
		else if (strcmp(k, "regularization") == 0) {
			c->regularization = str2regularization(tokens[1]);
		}
		else if (strcmp(k, "decay") == 0) {
			c->decay = str2float(tokens[1]);
		}
		else if (strcmp(k, "saturation") == 0) {
			c->saturation = tokens2floatarr(&tokens[1]);
		}
		else if (strcmp(k, "exposure") == 0) {
			c->exposure = tokens2floatarr(&tokens[1]);
		}
		else if (strcmp(k, "hue") == 0) {
			c->hue = tokens2floatarr(&tokens[1]);
		}
	}
}

void copy_to_layer_cfg(layer_cfg* l, char** tokens, network_cfg* c) {
	size_t t = tokens_length(tokens);
	if (t < 2) {
		printf("Error. Tokens must have a minimum length of 2, has %zu.\n", t);
		printf("Tokens %s\n", *tokens);
		wait_for_key_then_exit();
	}
	char* k = tokens[0];
	if (strcmp(k, "train") == 0) {
		l->train = str2int(tokens[1]);
	}
	else if (strcmp(k, "in_ids") == 0) {
		l->in_ids = tokens2intarr(&tokens[1]);
	}
	else if (strcmp(k, "chunk") == 0) {
		parse_chunk_args(l, tokens);
	}
	else if (strcmp(k, "batchnorm") == 0) {
		l->batchnorm = str2int(tokens[1]);
	}
	else if (strcmp(k, "filters") == 0 || strcmp(k, "outputs") == 0) {
		/*if (l->type == LAYER_DENSE) {
			return;
		}*/
		if (strcmp(tokens[1], "num_classes") == 0) {
			l->n_filters = (int)c->n_classes;
		}
		else if (strcmp(tokens[1], "detect") == 0) {
			l->n_filters = -1;
		}
		else {
			l->n_filters = str2int(tokens[1]);
		}
	}
	else if (strcmp(k, "groups") == 0) {
		l->n_groups = str2sizet(tokens[1]);
	}
	else if (strcmp(k, "kernel_size") == 0) {
		l->kernel_size = str2sizet(tokens[1]);
	}
	else if (strcmp(k, "stride") == 0) {
		l->stride = str2sizet(tokens[1]);
	}
	else if (strcmp(k, "pad") == 0) {
		l->pad = str2sizet(tokens[1]);
	}
	else if (strcmp(k, "activation") == 0) {
		l->activation = str2activation(tokens[1]);
	}
	else if (strcmp(k, "classes") == 0) {
		l->n_classes = str2sizet(tokens[1]);
	}
	else if (strcmp(k, "loss") == 0) {
		l->loss_type = str2loss(tokens[1]);
	}
	else if (strcmp(k, "anchors") == 0) {
		l->anchors = tokens2floatarr(&tokens[1]);
	}
	else if (strcmp(k, "objectness_smooth") == 0) {
		l->objectness_smooth = str2int(tokens[1]);
	}
}

void copy_cfg_to_network(network_cfg* net_cfg, network* net) {
	// [data]
	net->dataset_dir = net_cfg->dataset_dir;
	net->weights_file = net_cfg->weights_file;
	net->save_dir = net_cfg->save_dir;
	// [net]
	net->w = net_cfg->width;
	net->h = net_cfg->height;
	net->c = net_cfg->channels;
	net->n_classes = net_cfg->n_classes;
	net->use_gpu = net_cfg->use_gpu;
	// [training]
	net->batch_size = net_cfg->batch_size;
	net->subbatch_size = net_cfg->subbatch_size;
	net->max_iterations = net_cfg->max_iterations;
	net->save_frequency = net_cfg->save_frequency;
	net->learning_rate = net_cfg->learning_rate;
	net->lr_policy = net_cfg->lr_policy;
	net->step_percents = net_cfg->step_percents;
	net->step_scaling = net_cfg->step_scaling;
	net->poly_pow = net_cfg->poly_pow;
	net->coswr_frequency = net_cfg->coswr_frequency;
	net->coswr_multi = net_cfg->coswr_multi;
	net->ease_in = net_cfg->ease_in;
	net->momentum = net_cfg->momentum;
	net->regularization = net_cfg->regularization;
	net->decay = net_cfg->decay;
	copy_data_augment_range(net->saturation, net_cfg->saturation);
	copy_data_augment_range(net->exposure, net_cfg->exposure);
	copy_data_augment_range(net->hue, net_cfg->hue);
	// [layers]
	node* noed = net_cfg->layers->first;
	int n = (int)net_cfg->layers->length;
	for (int i = 0; i < n; i++) {
		assert(noed);
		layer* l = &net->layers[i];
		layer_cfg* l_cfg = (layer_cfg*)noed->val;
		assert(i == l_cfg->id);
		l->id = l_cfg->id;
		l->type = l_cfg->type;
		l->activation = l_cfg->activation;
		if (l_cfg->n_filters == -1) {
			if (i + 1 >= n) {
				printf("Invalid input for filters in layer %d\n", i);
				wait_for_key_then_exit();
			}
			assert(noed->next);
			layer_cfg* next_l_cfg = (layer_cfg*)noed->next->val;
			if (next_l_cfg->type != LAYER_DETECT) {
				printf("Filters cannot be set to 'detect' for a layer that does not output to a 'detect' layer.\n");
				wait_for_key_then_exit();
			}
			size_t n_anchors = next_l_cfg->anchors.n / 2;
			l->k.n_filters = (NUM_ANCHOR_PARAMS + net->n_classes) * n_anchors;
		}
		else {
			l->k.n_filters = l_cfg->n_filters;
		}

		l->n_inputs = max(l_cfg->in_ids.n, 1);
		if (l->id == 0 && l->n_inputs > 1) {
			printf("Error: First layer cannot have multiple inputs.\n");
			wait_for_key_then_exit();
		}
		l->inputs = (layer_input*)xcalloc(l->n_inputs, sizeof(layer_input));
		for (int j = 0; j < l->n_inputs; j++) {
			if (j < l_cfg->in_chunks.n) {
				l->inputs[j].n_chunks = l_cfg->in_chunks.a[j];
				l->inputs[j].chunk_idx = l_cfg->in_chunk_idxs[j];
			}
			if (j < l_cfg->in_ids.n) {
				if (l_cfg->in_ids.a[j] < 0) l_cfg->in_ids.a[j] += l->id;  // relative id handling
				if (l_cfg->in_ids.a[j] > l->id || l_cfg->in_ids.a[j] < 0) {
					printf("Error: Invalid input id of %d for layer %d\n", l_cfg->in_ids.a[j], l->id);
					wait_for_key_then_exit();
				}
				l->inputs[j].source_id = l_cfg->in_ids.a[j];
			}
			else {
				assert(j == 0);
				l->inputs[0].source_id = l->id - 1;
				break;
			}
		}
		
		l->k.n_groups = l_cfg->n_groups;
		l->k.ksize = l_cfg->kernel_size;
		l->k.pad = l_cfg->pad;
		l->k.stride = l_cfg->stride;
		l->batchnorm = l_cfg->batchnorm;
		l->train = l_cfg->train;
		l->loss_type = l_cfg->loss_type;
		l->n_classes = l_cfg->n_classes;
		l->objectness_smooth = l_cfg->objectness_smooth;
		if (l->type == LAYER_DETECT) {
			net->type = NET_DETECT;
			if (l_cfg->anchors.n % 2 != 0 || l_cfg->anchors.n == 0) {
				printf("Invalid input for anchors in layer %d. Each anchor must have two values.\n", i);
				wait_for_key_then_exit();
			}
			l->n_anchors = l_cfg->anchors.n / 2;
			l->anchors = (bbox*)xcalloc(l->n_anchors, sizeof(bbox));
			for (size_t j = 0; j < l->n_anchors; j++) {
				l->anchors[j].w = l_cfg->anchors.a[j * 2];
				l->anchors[j].h = l_cfg->anchors.a[j * 2 + 1];
			}
			net->n_anchors += l->n_anchors;
			xfree(&l_cfg->anchors.a);
		}
		else if (l->type == LAYER_CLASSIFY) net->type = NET_CLASSIFY;
		noed = noed->next;
	}
	load_class_names(net_cfg->classes_file, net, net_cfg->n_classes);
}

void copy_data_augment_range(float* dst, floatarr src) {
	float x = 1.0;  // default
	float y = 1.0;	// default
	if (src.n == 1) {
		if (src.a[0] > 1) {
			y = src.a[0];
		}
		else x = src.a[0];
	}
	else if (src.n > 1) {
		if (src.a[1] < src.a[0]) {
			x = src.a[1];
			y = src.a[0];
		}
	}
	dst[0] = x;
	dst[1] = y;
}

void load_class_names(char* filename, network* net, size_t n_classes) {
	if (!filename) {
		if (net->type == NET_CLASSIFY) return;
		printf("No classes file specified.\n");
		wait_for_key_then_exit();
	}
	FILE* file = get_filestream(filename, "r");
	char buff[LINESIZE] = { 0 };
	size_t n = get_line_count(file);
	if (n == 0) {
		printf("Classes file is empty. %s\n", filename);
		wait_for_key_then_exit();
	}
	else if (n != n_classes) {
		printf("Number of classes found in %s do not match # of classes in cfg file. (%zu =/= %zu)\n", filename, n, n_classes);
		wait_for_key_then_exit();
	}
	char** names = (char**)xcalloc(n, sizeof(char*));
	for (size_t i = 0; i < n; i++) {
		read_line_to_buff(file, buff, LINESIZE);
		clean_string(buff);
		size_t length = strlen(buff);
		names[i] = (char*)xcalloc(length + 1, sizeof(char));
		strcpy(names[i], buff);
	}
	net->class_names = names;
}

int is_header(char* str, char* header, int* is_layer) {
	if (str[0] != '[') return 0;
	if (strcmp(str, hDATA) == 0) {
		strcpy(header, hDATA);
		*is_layer = 0;
		return 1;
	}
	else if (strcmp(str, hNET) == 0) {
		strcpy(header, hNET);
		*is_layer = 0;
		return 1;
	}
	else if (strcmp(str, hTRAINING) == 0) {
		strcpy(header, hTRAINING);
		*is_layer = 0;
		return 1;
	}
	else if (strcmp(str, hCONV) == 0) {
		strcpy(header, hCONV);
		*is_layer = 1;
		return 1;
	}
	else if (strcmp(str, hCLASSIFY) == 0) {
		strcpy(header, hCLASSIFY);
		*is_layer = 1;
		return 1;
	}
	else if (strcmp(str, hMAXPOOL) == 0) {
		strcpy(header, hMAXPOOL);
		*is_layer = 1;
		return 1;
	}
	else if (strcmp(str, hDETECT) == 0) {
		strcpy(header, hDETECT);
		*is_layer = 1;
		return 1;
	}
	else if (strcmp(str, hDENSE) == 0) {
		strcpy(header, hDENSE);
		*is_layer = 1;
		return 1;
	}
	else if (strcmp(str, hRESIDUAL) == 0) {
		strcpy(header, hRESIDUAL);
		*is_layer = 1;
		return 1;
	}
	else if (strcmp(str, hAVGPOOL_GLOBAL) == 0) {
		strcpy(header, hAVGPOOL_GLOBAL);
		*is_layer = 1;
		return 1;
	}
	else if (strcmp(str, hUPSAMPLE) == 0) {
		strcpy(header, hUPSAMPLE);
		*is_layer = 1;
		return 1;
	}
	else if (strcmp(str, hROUTE) == 0) {
		strcpy(header, hROUTE);
		*is_layer = 1;
		return 1;
	}
	return 0;
}

#pragma warning(suppress:4715)  // Visual Studio being stupid (4175 = not all control paths return a value)
LOSS_TYPE str2loss(char* str) {
	if (strcmp(str, "mse") == 0) return LOSS_MSE;
	if (strcmp(str, "mae") == 0) return LOSS_MAE;
	if (strcmp(str, "cce") == 0) return LOSS_CCE;
	if (strcmp(str, "bce") == 0) return LOSS_BCE;
	printf("Error: No valid loss function named %s.\n", str);
	wait_for_key_then_exit();
}

#pragma warning(suppress:4715)  // Visual Studio being stupid (4175 = not all control paths return a value)
REGULARIZATION str2regularization(char* str) {
	char reg[3] = { 0, 0, 0 };
	reg[1] = str[1];
	reg[0] = (str[0] == 'L') ? 'l' : str[0];
	if (strcmp(reg, "l2") == 0) return REG_L2;
	if (strcmp(reg, "l1") == 0) return REG_L1;
	printf("Unknown regularization %s.\n", reg);
	wait_for_key_then_exit();
}

LR_POLICY str2lr_policy(char* str) {
	if (strcmp(str, "steps") == 0) {
		return LR_STEPS;
	}
	if (strcmp(str, "poly") == 0) {
		return LR_POLYNOMIAL;
	}
	if (strcmp(str, "exp") == 0) {
		return LR_EXPONENTIAL;
	}
	fprintf(stderr, "Error: No valid policy named %s.\n", str);
	exit(EXIT_FAILURE);
}

ACTIVATION str2activation(char* str) {
	if (strcmp(str, "relu") == 0) return ACT_RELU;
	if (strcmp(str, "mish") == 0) return ACT_MISH;
	if (strcmp(str, "leaky") == 0) return ACT_LEAKY;
	if (strcmp(str, "sigmoid") == 0) return ACT_SIGMOID;
	if (strcmp(str, "softmax") == 0) return ACT_SOFTMAX;
	if (strcmp(str, "tanh") == 0) return ACT_TANH;
	if (strcmp(str, "none") == 0) return ACT_NONE;
	if (strcmp(str, "linear") == 0) return ACT_NONE;
	fprintf(stderr, "Error: No valid activation named %s.\n", str);
	exit(EXIT_FAILURE);
}

floatarr tokens2floatarr(char** tokens) {
	size_t n = tokens_length(tokens);
	floatarr farr = { 0 };
	farr.n = n;
	farr.a = (float*)xcalloc(n, sizeof(float));
	for (size_t i = 0; i < n; i++) {
		farr.a[i] = str2float(tokens[i]);
	}
	return farr;
}

intarr tokens2intarr(char** tokens) {
	size_t n = tokens_length(tokens);
	intarr iarr = { 0 };
	iarr.n = n;
	iarr.a = (int*)xcalloc(n, sizeof(int));
	for (size_t i = 0; i < n; i++) {
		iarr.a[i] = str2int(tokens[i]);
	}
	return iarr;
}

void parse_chunk_args(layer_cfg* l, char** tokens) {
	int chunk_idx;
	int n_chunks;
	size_t n = tokens_length(tokens);
	l->in_chunks.n = n;
	l->in_chunks.a = (int*)xcalloc(n, sizeof(int));
	l->in_chunk_idxs = (int*)xcalloc(n, sizeof(int));
	for (size_t i = 0; i < n; i++) {
		char** args = split_string(tokens[i], ":");
		if (tokens_length(args) < 2) {
			printf("Error: Invalid argument for chunk. Layer ID: %d\n", l->id);
			wait_for_key_then_exit();
		}
		chunk_idx = str2int(args[0]);
		n_chunks = str2int(args[1]);
		if (chunk_idx > n_chunks) {
			printf("Error: Chunk index cannot be greater than # of chunks. Layer ID: %d\n", l->id);
			wait_for_key_then_exit();
		}
		if (chunk_idx < 1) {
			printf("Error: Chunk index cannot be less than 1. Layer ID: %d\n", l->id);
			wait_for_key_then_exit();
		}
		chunk_idx--;
		l->in_chunks.a[i] = n_chunks;
		l->in_chunk_idxs[i] = chunk_idx;
		xfree(args);
	}
}

LAYER_TYPE header2layertype(char* header) {
	if (strcmp(header, hCONV) == 0) return LAYER_CONV;
	if (strcmp(header, hCLASSIFY) == 0) return LAYER_CLASSIFY;
	if (strcmp(header, hMAXPOOL) == 0) return LAYER_MAXPOOL;
	if (strcmp(header, hDETECT) == 0) return LAYER_DETECT;
	if (strcmp(header, hDENSE) == 0) return LAYER_DENSE;
	if (strcmp(header, hRESIDUAL) == 0) return LAYER_RESIDUAL;
	if (strcmp(header, hAVGPOOL_GLOBAL) == 0) return LAYER_AVGPOOL_GLOBAL;
	if (strcmp(header, hUPSAMPLE) == 0) return LAYER_UPSAMPLE;
	if (strcmp(header, hROUTE) == 0) return LAYER_ROUTE;
	return LAYER_NONE;
}

void print_network_cfg(network_cfg* c) {
	printf("\n[CFG]\n\n");

	printf("[DATA]\n");
	printf("dataset_dir = %s\n", c->dataset_dir);
	printf("classes_file = %s\n", c->classes_file);
	printf("weights_file = %s\n", c->weights_file);
	printf("backup_dir = %s\n", c->save_dir);

	printf("\n[NET]\n");
	printf("width = %zu\n", c->width);
	printf("height = %zu\n", c->height);
	printf("channels = %zu\n", c->channels);
	printf("n_classes = %zu\n", c->n_classes);

	printf("\n[TRAINING]\n");
	printf("batch_size = %zu\n", c->batch_size);
	printf("subbatch_size = %zu\n", c->subbatch_size);
	printf("max_iterations = %zu\n", c->max_iterations);
	printf("learning_rate = %f\n", c->learning_rate);
	printf("lr_policy = ");
	print_lrpolicy(c->lr_policy);
	printf("step_percents = ");
	print_floatarr(&c->step_percents);
	printf("step_scaling = ");
	print_floatarr(&c->step_scaling);
	printf("ease_in = %zu\n", c->ease_in);
	printf("momentum = %f\n", c->momentum);
	printf("decay = %f\n", c->decay);
	printf("saturation = ");
	print_floatarr(&c->saturation);
	printf("exposure = ");
	print_floatarr(&c->exposure);
	printf("hue = ");
	print_floatarr(&c->hue);

	printf("\n\n[LAYERS]\n");
	printf("# of layers = %zu\n\n", c->layers->length);
	node* n = c->layers->first;
	node* next;
	while (n) {
		next = n->next;
		print_layer_cfg((layer_cfg*)n->val);
		printf("\n");
		n = next;
	}
	printf("[END CFG]\n\n");
}

void print_layer_cfg(layer_cfg* l) {
	if (l->type == LAYER_CONV) printf(hCONV);
	else if (l->type == LAYER_CLASSIFY) printf(hCLASSIFY);
	else if (l->type == LAYER_MAXPOOL) printf(hMAXPOOL);
	else if (l->type == LAYER_DETECT) printf(hDETECT);
	else if (l->type == LAYER_DENSE) printf(hDENSE);
	else if (l->type == LAYER_RESIDUAL) printf(hRESIDUAL);
	else if (l->type == LAYER_AVGPOOL_GLOBAL) printf(hAVGPOOL_GLOBAL);
	else if (l->type == LAYER_UPSAMPLE) printf(hUPSAMPLE);
	printf("\nid = %d\n", l->id);
	printf("train = %d\n", l->train);
	printf("in_ids = ");
	print_intarr(&l->in_ids);
	printf("batchnorm = %d\n", l->batchnorm);
	printf("n_filters = %d\n", l->n_filters);
	printf("kernel_size = %zu\n", l->kernel_size);
	printf("stride = %zu\n", l->stride);
	printf("pad = %zu\n", l->pad);
	printf("activation = ");
	print_activation(l->activation);
	printf("n_classes = %zu\n", l->n_classes);
	printf("loss = ");
	print_loss_type(l->loss_type);
}