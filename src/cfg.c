#include "cfg.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "xallocs.h"


#define hDATA "[data]"
#define hNET "[net]"
#define hTRAINING "[training]"
#define hCONV "[conv]"
#define hFC "[fc]"
#define hMAXPOOL "[maxpool]"
#define hRESIDUAL "[residual]"
#define hCLASSIFY "[classify]"
#define hDETECT "[detect]"
#define hAVGPOOL "[avgpool]"
#define hUPSAMPLE "[upsample]"

#define LINESIZE 512
#define TOKENSSIZE 64


void load_cfg(char* filename, cfg* c);
void copy_to_cfg(cfg* c, char** tokens, char* header);
void copy_to_cfg_layer(cfg_layer* l, char** tokens, cfg* c);
void copy_cfg_to_network(cfg* cfig, network* net);
void copy_data_augment_range(float* dst, floatarr src);
void load_class_names(char* filename, network* net, size_t n_classes);
LR_POLICY str2lr_policy(char* str);
ACTIVATION str2activation(char* str);
LOSS_TYPE str2loss(char* str);
REGULARIZATION str2regularization(char* str);
floatarr tokens2floatarr(char** tokens, size_t offset);
intarr tokens2intarr(char** tokens, size_t offset);
int is_header(char* str, char* header, int* is_layer);
LAYER_TYPE header2layertype(char* header);
void print_cfg(cfg* c);
void print_cfg_layer(cfg_layer* l);


network* create_network_from_cfg(char* cfgfile) {
	cfg c = { 0 };
	printf("Loading cfg file: %s\n", cfgfile);
	load_cfg(cfgfile, &c);
	if (c.layers == NULL) {
		printf("No layers in cfg.\n");
		wait_for_key_then_exit();
	}
#pragma warning (suppress:6011)  // Visual Studio is too stupid to recognize c.layers can't be null here
	network* net = new_network(c.layers->length);
	copy_cfg_to_network(&c, net);
	net->cfg_file = cfgfile;
	xfree(c.classes_file);
	xfree(c.saturation.a);
	xfree(c.exposure.a);
	xfree(c.hue.a);
	free_list(c.layers, 1);
	build_network(net);
	return net;
}

void load_cfg(char* filename, cfg* c) {
	FILE* file = get_filestream(filename, "r");
	char line[LINESIZE] = { 0 };
	char* tokens[TOKENSSIZE] = { 0 };
	char header[50] = { 0 };
	cfg_layer* l = 0;
	list* layers = (list*)xcalloc(1, sizeof(list));
	int is_layer = 0;
	while (read_line_to_buff(file, line, LINESIZE)) {
		clean_string(line);
		split_string_to_buff(line, "=,", tokens);
		if (tokens[0] == NULL) continue;
		if (is_header(line, header, &is_layer)) {
			if (!is_layer) continue;
			if (l) c->batchnorm = (l->batchnorm) ? 1 : 0;
			l = (cfg_layer*)xcalloc(1, sizeof(cfg_layer));
			l->train = 1;  // default
			list_append(layers, l);
			l->type = header2layertype(header);
		}
		else if (!is_layer) {
			copy_to_cfg(c, tokens, header);
		}
		else {
			copy_to_cfg_layer(l, tokens, c);
		}
	}
	c->layers = layers;
	close_filestream(file);
}

void copy_to_cfg(cfg* c, char** tokens, char* header) {
	size_t t = tokens_length(tokens);
	if (t < 2) {
		printf("Error. Tokens must have a minimum length of 2, has %zu.\n", t);
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
			c->step_percents = tokens2floatarr(tokens, 1);
		}
		else if (strcmp(k, "step_scaling") == 0) {
			c->step_scaling = tokens2floatarr(tokens, 1);
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
			c->saturation = tokens2floatarr(tokens, 1);
		}
		else if (strcmp(k, "exposure") == 0) {
			c->exposure = tokens2floatarr(tokens, 1);
		}
		else if (strcmp(k, "hue") == 0) {
			c->hue = tokens2floatarr(tokens, 1);
		}
	}
}

void copy_to_cfg_layer(cfg_layer* l, char** tokens, cfg* c) {
	size_t t = tokens_length(tokens);
	if (t < 2) {
		printf("Error. Tokens must have a minimum length of 2, has %zu.\n", t);
		wait_for_key_then_exit();
	}
	char* k = tokens[0];
	if (strcmp(k, "id") == 0) {
		l->id = str2int(tokens[1]);
	}
	else if (strcmp(k, "train") == 0) {
		l->train = str2int(tokens[1]);
	}
	else if (strcmp(k, "in_ids") == 0) {
		l->in_ids = tokens2intarr(tokens, 1);
	}
	else if (strcmp(k, "batchnorm") == 0) {
		l->batchnorm = str2int(tokens[1]);
	}
	else if (strcmp(k, "filters") == 0) {
		if (strcmp(tokens[1], "num_classes") == 0) {
			l->n_filters = c->n_classes;
		}
		else if (strcmp(tokens[1], "anchors") == 0) {
			l->n_filters = (NUM_ANCHOR_PARAMS + c->n_classes) * 3;
		}
		else {
			l->n_filters = str2sizet(tokens[1]);
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
		l->anchors = tokens2floatarr(tokens, 1);
	}
}

void copy_cfg_to_network(cfg* cfig, network* net) {
	// [data]
	net->dataset_dir = cfig->dataset_dir;
	net->weights_file = cfig->weights_file;
	net->save_dir = cfig->save_dir;
	// [net]
	net->w = cfig->width;
	net->h = cfig->height;
	net->c = cfig->channels;
	net->n_classes = cfig->n_classes;
	net->batchnorm = cfig->batchnorm;
	// [training]
	net->batch_size = cfig->batch_size;
	net->subbatch_size = cfig->subbatch_size;
	net->max_iterations = cfig->max_iterations;
	net->save_frequency = cfig->save_frequency;
	net->learning_rate = cfig->learning_rate;
	net->lr_policy = cfig->lr_policy;
	net->step_percents = cfig->step_percents;
	net->step_scaling = cfig->step_scaling;
	net->poly_pow = cfig->poly_pow;
	net->coswr_frequency = cfig->coswr_frequency;
	net->coswr_multi = cfig->coswr_multi;
	net->ease_in = cfig->ease_in;
	net->momentum = cfig->momentum;
	net->regularization = cfig->regularization;
	net->decay = cfig->decay;
	copy_data_augment_range(net->saturation, cfig->saturation);
	copy_data_augment_range(net->exposure, cfig->exposure);
	copy_data_augment_range(net->hue, cfig->hue);
	// [layers]
	node* noed = cfig->layers->first;
	int n = (int)cfig->layers->length;
	for (int i = 0; i < n; i++) {
		assert(noed);
		layer* l = &net->layers[i];
		cfg_layer* cl = (cfg_layer*)noed->val;
		cl->id = i;
		l->id = cl->id;
		l->type = cl->type;
		l->activation = cl->activation;
		l->n_filters = cl->n_filters;
		l->n_groups = cl->n_groups;
		l->batchnorm = cl->batchnorm;
		l->ksize = cl->kernel_size;
		l->in_ids = cl->in_ids;
		l->pad = cl->pad;
		l->stride = cl->stride;
		l->train = cl->train;
		l->loss_type = cl->loss_type;
		l->n_classes = cl->n_classes;
		if (l->type == LAYER_DETECT) {
			net->type = NET_DETECT;
			if (cl->anchors.n % 2 != 0 || cl->anchors.n == 0) {
				printf("Invalid input for anchors in layer %d. Each anchor must have two values.\n", i);
				wait_for_key_then_exit();
			}
			l->n_anchors = cl->anchors.n / 2;
			l->anchors = (bbox*)xcalloc(l->n_anchors, sizeof(bbox));
			for (size_t j = 0; j < l->n_anchors; j++) {
				l->anchors[j].w = cl->anchors.a[j * 2];
				l->anchors[j].h = cl->anchors.a[j * 2 + 1];
			}
			net->n_anchors += l->n_anchors;
			xfree(cl->anchors.a);
		}
		else if (l->type == LAYER_CLASSIFY) net->type = NET_CLASSIFY;
		noed = noed->next;
	}
	load_class_names(cfig->classes_file, net, cfig->n_classes);
}

void copy_data_augment_range(float* dst, floatarr src) {
	float x = 1.0;
	float y = 1.0;
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
	else if (strcmp(str, hFC) == 0) {
		strcpy(header, hFC);
		*is_layer = 1;
		return 1;
	}
	else if (strcmp(str, hRESIDUAL) == 0) {
		strcpy(header, hRESIDUAL);
		*is_layer = 1;
		return 1;
	}
	else if (strcmp(str, hAVGPOOL) == 0) {
		strcpy(header, hAVGPOOL);
		*is_layer = 1;
		return 1;
	}
	else if (strcmp(str, hUPSAMPLE) == 0) {
		strcpy(header, hUPSAMPLE);
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
	fprintf(stderr, "Error: No valid activation named %s.\n", str);
	exit(EXIT_FAILURE);
}

floatarr tokens2floatarr(char** tokens, size_t offset) {
	size_t n = tokens_length(tokens);
	assert(n > offset);
	floatarr farr = { 0 };
	size_t length = n - offset;
	farr.n = length;
	farr.a = (float*)xcalloc(length, sizeof(float));
	for (size_t i = 0; i < length; i++) {
		farr.a[i] = str2float(tokens[i + offset]);
	}
	return farr;
}

intarr tokens2intarr(char** tokens, size_t offset) {
	size_t n = tokens_length(tokens);
	assert(n > offset);
	intarr iarr = { 0 };
	size_t length = n - offset;
	iarr.n = length;
	iarr.a = (int*)xcalloc(length, sizeof(int));
	for (size_t i = 0; i < length; i++) {
		iarr.a[i] = str2int(tokens[i + offset]);
	}
	return iarr;
}

LAYER_TYPE header2layertype(char* header) {
	if (strcmp(header, hCONV) == 0) return LAYER_CONV;
	if (strcmp(header, hCLASSIFY) == 0) return LAYER_CLASSIFY;
	if (strcmp(header, hMAXPOOL) == 0) return LAYER_MAXPOOL;
	if (strcmp(header, hDETECT) == 0) return LAYER_DETECT;
	if (strcmp(header, hFC) == 0) return LAYER_FC;
	if (strcmp(header, hRESIDUAL) == 0) return LAYER_RESIDUAL;
	if (strcmp(header, hAVGPOOL) == 0) return LAYER_AVGPOOL;
	if (strcmp(header, hUPSAMPLE) == 0) return LAYER_UPSAMPLE;
	return LAYER_NONE;
}

void print_cfg(cfg* c) {
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
		print_cfg_layer((cfg_layer*)n->val);
		printf("\n");
		n = next;
	}
	printf("[END CFG]\n\n");
}

void print_cfg_layer(cfg_layer* l) {
	if (l->type == LAYER_CONV) printf(hCONV);
	else if (l->type == LAYER_CLASSIFY) printf(hCLASSIFY);
	else if (l->type == LAYER_MAXPOOL) printf(hMAXPOOL);
	else if (l->type == LAYER_DETECT) printf(hDETECT);
	else if (l->type == LAYER_FC) printf(hFC);
	else if (l->type == LAYER_RESIDUAL) printf(hRESIDUAL);
	else if (l->type == LAYER_AVGPOOL) printf(hAVGPOOL);
	else if (l->type == LAYER_UPSAMPLE) printf(hUPSAMPLE);
	printf("\nid = %d\n", l->id);
	printf("train = %d\n", l->train);
	printf("in_ids = ");
	print_intarr(&l->in_ids);
	printf("out_ids = ");
	print_intarr(&l->out_ids);
	printf("batchnorm = %d\n", l->batchnorm);
	printf("n_filters = %zu\n", l->n_filters);
	printf("kernel_size = %zu\n", l->kernel_size);
	printf("stride = %zu\n", l->stride);
	printf("pad = %zu\n", l->pad);
	printf("activation = ");
	print_activation(l->activation);
	printf("n_classes = %zu\n", l->n_classes);
	printf("loss = ");
	print_loss_type(l->loss_type);
}