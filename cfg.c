#include "cfg.h"
#include "list.h"
#include "xallocs.h"
#include "utils.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>


list* get_cfg_sections(char* filename, size_t* n_layers);
void append_cfg_section(list* lst, char* header);
cfg_data* new_cfg_data(void);
cfg_net* new_cfg_net(void);
cfg_training* new_cfg_training(void);
cfg_conv* new_cfg_conv(void);
cfg_classify* new_cfg_classify(void);
void set_param_cfg_data(cfg_section* section, char** tokens);
void set_param_cfg_net(cfg_section* section, char** tokens);
void set_param_cfg_training(cfg_section* section, char** tokens);
void set_param_cfg_conv(cfg_section* section, char** tokens);
void set_param_cfg_classify(cfg_section* section, char** tokens);
LR_POLICY str2lr_policy(char* str);
ACTIVATION str2activation(char* str);
COST_TYPE str2cost(char* str);
floatarr tokens2floatarr(char** tokens, size_t offset);
intarr tokens2intarr(char** tokens, size_t offset);
void print_tokens(char** tokens);
void print_cfg_net(cfg_net* s);
void print_cfg_training(cfg_training* s);
void print_cfg_conv(cfg_conv* s);
void print_cfg_classify(cfg_classify* s);
void print_cfg(list* sections);
void print_tokens(char** tokens);
LAYER_TYPE header2layertype(char* header);
int is_layer_header(char* header);
void load_cfg_to_network(list* sections, network* net);
void load_cfg_net_to_network(cfg_net* s, network* n);
void load_cfg_training_to_network(cfg_training* s, network* n);
int load_cfg_section_to_layer(cfg_section* s, network* n);
void load_cfg_conv_to_layer(cfg_conv* s, layer* l);
void load_cfg_classify_to_layer(cfg_classify* s, layer* l);
void free_sections(list* lst);
void free_section(cfg_section* s);
void free_cfg_data(cfg_data* s);
void free_cfg_net(cfg_net* s);
void free_cfg_training(cfg_training* s);
void free_cfg_conv(cfg_conv* s);
void free_cfg_classify(cfg_classify* s);


static char* headers[] = { "[data]", "[net]", "[training]", "[conv]", "[classify]", "\0"};


network* create_network_from_cfg(char* cfgfile) {
	size_t n_layers;
	list* sections = get_cfg_sections(cfgfile, &n_layers);
	network* net = new_network(n_layers);
	print_cfg(sections);
	load_cfg_to_network(sections, net);
	free_sections(sections);
	build_network(net);
	return net;
}

list* get_cfg_sections(char* filename, size_t* n_layers) {
	*n_layers = 0;
	FILE* file = get_filestream(filename, "r");
	list* sections = new_list();
	cfg_section* section;
	int h = -1;  // index of headers[] corresponding to current header being read
	char* line;
	while (line = read_line(file), line != 0) {
		clean_string(line);
		char** tokens = split_string(line, "=,");
		if (tokens == NULL) {
			xfree(line);
			continue;
		}
		if (tokens[0][0] == '[') {
			int i = 0;
			while (headers[i][0] != '\0') {
				if (strcmp(tokens[0], headers[i]) == 0) {
					h = i;
					printf("Appending cfg section: %s\n", headers[h]);
					append_cfg_section(sections, headers[h]);
					if (is_layer_header(headers[h])) (*n_layers)++;
					break;
				}
				i++;
			}
			if (headers[i][0] == '\0') printf("Unknown header %s.\n", tokens[0]);
		}
		else if (h > -1) {
			section = (cfg_section*)(sections->last->val);
			section->set_param(section, tokens);
		}
		xfree(line);
		xfree(tokens);
	}
	close_filestream(file);
	//print_cfg(sections);
	return sections;
}

void load_cfg_to_network(list* sections, network* net) {
	node* noed = sections->first;
	cfg_section* sec;
	size_t i = 0;
	while (noed != NULL) {
		sec = (cfg_section*)noed->val;
		if (strcmp(sec->header, "[net]") == 0) {
			load_cfg_net_to_network((cfg_net*)sec, net);
		}
		else if (strcmp(sec->header, "[training]") == 0) {
			load_cfg_training_to_network((cfg_training*)sec, net);
		}
		else if (load_cfg_section_to_layer(sec, net)) {
			i++;
		}
		noed = noed->next;
	}
	assert(net->n_layers == i); // number of layers counted equals number of layers loaded
}

void load_cfg_data_to_network(cfg_data* s, network* net) {
	net->dataset_dir = s->dataset_dir;
	net->class_names = load_class_names(s->classes_file);
	net->weights_file = s->weights_file;
	net->backup_dir = s->backup_dir;
}

char** load_class_names(char* classes_file) {
	FILE* file = get_filestream(classes_file, "r");
	char* line;
	size_t n = 0;
	while (1) {
		line = read_line(file);
		if (!line) break;
		clean_string(line);
		if (!strlen(line)) {
			printf("Invalid line in classes file.\nLine index %zu\n", n);
		}
		n++;
	}
}

void load_cfg_net_to_network(cfg_net* s, network* net) {
	net->w = s->width;
	net->h = s->height;
	net->c = s->channels;
	net->n_classes = s->num_classes;
	net->cost = s->cost;
}

void load_cfg_training_to_network(cfg_training* s, network* n) {
	n->batch_size = s->batch_size;
	n->subbatch_size = s->subbatch_size;
	n->max_iterations = s->max_iterations;
	n->learning_rate = s->learning_rate;
	n->lr_policy = s->lr_policy;
	n->step_percents = s->step_percents;
	n->step_scaling = s->step_scaling;
	n->ease_in = s->ease_in;
	n->momentum = s->momentum;
	n->decay = s->decay;
	n->saturation[0] = 1.0;
	n->saturation[1] = 1.0;
	if (s->saturation.n > 0) {
		n->saturation[0] = s->saturation.a[0];
		if (s->saturation.n == 2) {
			n->saturation[1] = s->saturation.a[1];
		}
	}
	n->exposure[0] = 1.0;
	n->exposure[1] = 1.0;
	if (s->exposure.n > 0) {
		n->exposure[0] = s->saturation.a[0];
		if (s->saturation.n == 2) {
			n->saturation[1] = s->saturation.a[1];
		}
	}
	n->hue[0] = 1.0;
	n->hue[1] = 1.0;
	if (s->hue.n > 0) {
		n->hue[0] = s->hue.a[0];
		if (s->hue.n == 2) {
			n->hue[1] = s->hue.a[1];
		}
	}
}

int load_cfg_section_to_layer(cfg_section* sec, network* net) {
	LAYER_TYPE ltype = header2layertype(sec->header);
	if (ltype == NONE_LAYER) return 0;
	int id = ((cfg_layer*)sec)->id;
	layer* l = &(net->layers[id]);
	if (l->type != (LAYER_TYPE)0) {
		printf("Layer ID %d already written to. Check for layers with same ID in cfg file.\n", l->id);
		exit(EXIT_FAILURE);
	}
	if (ltype == CONV) {
		load_cfg_conv_to_layer((cfg_conv*)sec, l);
		return 1;
	}
	if (ltype == CLASSIFY) {
		load_cfg_classify_to_layer((cfg_classify*)sec, l);
		return 1;
	}
	return 0;
}

void load_cfg_conv_to_layer(cfg_conv* s, layer* l) {
	l->type = CONV;
	l->id = s->id;
	l->batch_norm = s->batch_normalize;
	l->n_filters = s->n_filters;
	l->ksize = s->kernel_size;
	l->stride = s->stride;
	l->pad = s->pad;
	l->activation = s->activation;
	l->train = s->train;
	l->in_ids = s->in_ids;
	l->out_ids = s->out_ids;
}

void load_cfg_classify_to_layer(cfg_classify* s, layer* l) {
	l->type = CLASSIFY;
	l->id = s->id;
	l->train = s->train;
	l->n_filters = s->num_classes;
	l->cost = s->cost;
}

void append_cfg_section(list* lst, char* header) {
	if (strcmp(header, headers[0]) == 0) list_append(lst, new_cfg_data());
	else if (strcmp(header, headers[1]) == 0) list_append(lst, new_cfg_net());
	else if (strcmp(header, headers[2]) == 0) list_append(lst, new_cfg_training());
	else if (strcmp(header, headers[3]) == 0) list_append(lst, new_cfg_conv());
	else if (strcmp(header, headers[4]) == 0) list_append(lst, new_cfg_classify());
}

cfg_data* new_cfg_data(void) {
	cfg_data* section = (cfg_data*)xcalloc(1, sizeof(cfg_data));
	section->header = headers[0];
	section->set_param = set_param_cfg_data;
	return section;
}

cfg_net* new_cfg_net(void) {
	cfg_net* section = (cfg_net*)xcalloc(1, sizeof(cfg_net));
	section->header = headers[1];
	section->set_param = set_param_cfg_net;
	return section;
}

cfg_training* new_cfg_training(void) {
	cfg_training* section = (cfg_training*)xcalloc(1, sizeof(cfg_training));
	section->header = headers[2];
	section->set_param = set_param_cfg_training;
	return section;
}

cfg_conv* new_cfg_conv(void) {
	cfg_conv* section = (cfg_conv*)xcalloc(1, sizeof(cfg_conv));
	section->header = headers[3];
	section->set_param = set_param_cfg_conv;
	// Set non-zero defaults
	section->train = 1;
	section->stride = 1;
	return section;
}

cfg_classify* new_cfg_classify(void) {
	cfg_classify* section = (cfg_classify*)xcalloc(1, sizeof(cfg_classify));
	section->header = headers[4];
	section->set_param = set_param_cfg_classify;
	// Set non-zero defaults
	section->train = 1;
	return section;
}

void set_param_cfg_data(cfg_section* section, char** tokens) {
	cfg_data* sec = (cfg_data*)section;
	char* param = tokens[0];  //pointer to name of param
	if (strcmp(param, "dataset_dir") == 0) sec->dataset_dir = str2sizet(tokens[1]);
	else if (strcmp(param, "classes_file") == 0) sec->classes_file = str2sizet(tokens[1]);
	else if (strcmp(param, "weights_file") == 0) sec->weights_file = str2sizet(tokens[1]);
	else if (strcmp(param, "backup_dir") == 0) sec->backup_dir = str2sizet(tokens[1]);
	else {
		fprintf(stderr, "Error: No parameter named %s in section %s.\n", param, sec->header);
		exit(EXIT_FAILURE);
	}
}

void set_param_cfg_net(cfg_section* section, char** tokens) {
	cfg_net* sec = (cfg_net*)section;
	char* param = tokens[0];  //pointer to name of param
	if (strcmp(param, "width") == 0) sec->width = str2sizet(tokens[1]);
	else if (strcmp(param, "height") == 0) sec->height = str2sizet(tokens[1]);
	else if (strcmp(param, "channels") == 0) sec->channels = str2sizet(tokens[1]);
	else if (strcmp(param, "num_classes") == 0) sec->num_classes = str2sizet(tokens[1]);
	else if (strcmp(param, "cost") == 0) sec->cost = str2cost(tokens[1]);
	else {
		fprintf(stderr, "Error: No parameter named %s in section %s.\n", param, sec->header);
		exit(EXIT_FAILURE);
	}
}

void set_param_cfg_training(cfg_section* section, char** tokens) {
	cfg_training* sec = (cfg_training*)section;
	char* param = tokens[0];
	if (strcmp(param, "batch_size") == 0) sec->batch_size = str2sizet(tokens[1]);
	else if (strcmp(param, "subbatch_size") == 0) sec->subbatch_size = str2sizet(tokens[1]);
	else if (strcmp(param, "max_iterations") == 0) sec->max_iterations = str2sizet(tokens[1]);
	else if (strcmp(param, "learning_rate") == 0) sec->learning_rate = str2float(tokens[1]);
	else if (strcmp(param, "lr_policy") == 0) sec->lr_policy = str2lr_policy(tokens[1]);
	else if (strcmp(param, "step_percents") == 0) sec->step_percents = tokens2floatarr(tokens, 1);
	else if (strcmp(param, "step_scaling") == 0) sec->step_scaling = tokens2floatarr(tokens, 1);
	else if (strcmp(param, "ease_in") == 0) sec->ease_in = str2sizet(tokens[1]);
	else if (strcmp(param, "momentum") == 0) sec->momentum = str2float(tokens[1]);
	else if (strcmp(param, "decay") == 0) sec->decay = str2float(tokens[1]);
	else if (strcmp(param, "saturation") == 0) sec->saturation = tokens2floatarr(tokens, 1);
	else if (strcmp(param, "exposure") == 0) sec->exposure = tokens2floatarr(tokens, 1);
	else if (strcmp(param, "hue") == 0) sec->hue = tokens2floatarr(tokens, 1);
}

void set_param_cfg_conv(cfg_section* section, char** tokens) {
	cfg_conv* sec = (cfg_conv*)section;
	char* param = tokens[0];
	if (strcmp(param, "id") == 0) sec->id = str2int(tokens[1]);
	else if (strcmp(param, "batch_normalize") == 0) sec->batch_normalize = str2int(tokens[1]);
	else if (strcmp(param, "train") == 0) sec->train = str2int(tokens[1]);
	else if (strcmp(param, "filters") == 0) sec->n_filters = str2sizet(tokens[1]);
	else if (strcmp(param, "kernel_size") == 0) sec->kernel_size = str2sizet(tokens[1]);
	else if (strcmp(param, "stride") == 0) sec->stride = str2sizet(tokens[1]);
	else if (strcmp(param, "pad") == 0) sec->pad = str2sizet(tokens[1]);
	else if (strcmp(param, "activation") == 0) sec->activation = str2activation(tokens[1]);
	else if (strcmp(param, "in_ids") == 0) sec->in_ids = tokens2intarr(tokens, 1);
	else if (strcmp(param, "out_ids") == 0) sec->out_ids = tokens2intarr(tokens, 1);
}

void set_param_cfg_classify(cfg_section* section, char** tokens) {
	cfg_classify* sec = (cfg_classify*)section;
	char* param = tokens[0];
	if (strcmp(param, "id") == 0) sec->id = str2int(tokens[1]);
	else if (strcmp(param, "in_ids") == 0) sec->in_ids = tokens2intarr(tokens, 1);
	else if (strcmp(param, "train") == 0) sec->train = str2int(tokens[1]);
	else if (strcmp(param, "num_classes") == 0) sec->num_classes = str2sizet(param);
	else if (strcmp(param, "cost") == 0) sec->cost = str2cost(tokens[1]);
}

void free_sections(list* l) {
	node* n = l->first;
	while (n) {
		free_section((cfg_section*)n->val);
		n = n->next;
	}
	free_list(l);
}

void free_section(cfg_section* sec) {
	if (strcmp(sec->header, headers[0]) == 0) free_cfg_data((cfg_data*)sec);
	if (strcmp(sec->header, headers[1]) == 0) free_cfg_net((cfg_net*)sec);
	else if (strcmp(sec->header, headers[2]) == 0) free_cfg_training((cfg_training*)sec);
	else if (strcmp(sec->header, headers[3]) == 0) free_cfg_conv((cfg_conv*)sec);
	else if (strcmp(sec->header, headers[4]) == 0) free_cfg_classify((cfg_classify*)sec);
}

void free_cfg_data(cfg_data* s) {
	/*xfree(s->dataset_dir);  keep around to be stored in network
	xfree(s->weights_file);
	xfree(s->backup_dir);*/
	xfree(s->classes_file);
	xfree(s);
}

void free_cfg_net(cfg_net* s) {
	xfree(s);
}

void free_cfg_training(cfg_training* s) {
	/*xfree(s->step_percents.a);  keep around to be stored in layer
	xfree(s->step_scaling.a);*/
	xfree(s->saturation.a);
	xfree(s->exposure.a);
	xfree(s->hue.a);
	xfree(s);
}

void free_cfg_conv(cfg_conv* s) {
	/*xfree(s->in_ids.a);  keep around to be stored in layer
	xfree(s->out_ids.a);*/
	xfree(s);
}

void free_cfg_classify(cfg_classify* s) {
	xfree(s);
}

int is_layer_header(char* header) {
	if (strcmp(header, "[conv]") == 0) return 1;
	if (strcmp(header, "[classify]") == 0) return 1;
	else return 0;
}

LAYER_TYPE header2layertype(char* header) {
	if (strcmp(header, "[conv]") == 0) return CONV;
	if (strcmp(header, "[classify]") == 0) return CLASSIFY;
	return NONE_LAYER;
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

LR_POLICY str2lr_policy(char* str) {
	if (strcmp(str, "steps") == 0) {
		return LR_STEPS;
	}
	fprintf(stderr, "Error: No valid policy named %s.\n", str);
	exit(EXIT_FAILURE);
}

ACTIVATION str2activation(char* str) {
	if (strcmp(str, "relu") == 0) return RELU;
	if (strcmp(str, "mish") == 0) return MISH;
	if (strcmp(str, "logistic") == 0) return LOGISTIC;
	fprintf(stderr, "Error: No valid activation named %s.\n", str);
	exit(EXIT_FAILURE);
}

COST_TYPE str2cost(char* str) {
	if (strcmp(str, "mse") == 0) return MSE;
	if (strcmp(str, "bce") == 0) return BCE;
	if (strcmp(str, "cce") == 0) return CCE;
	fprintf(stderr, "Error: No valid cost function named %s.\n", str);
	exit(EXIT_FAILURE);
}

void print_tokens(char** tokens) {
	size_t i = 0;
	while (tokens[i] != NULL) {
		printf("token %zu = %s\n", i, tokens[i]);
		i++;
	}
}

void print_cfg(list* sections) {
	printf("\n[CFG]\n");
	size_t n = sections->length;
	printf("# of sections = %zu\n", n);
	cfg_section* section;
	for (int i = 0; i < n; i++) {
		section = (cfg_section*)list_get_item(sections, i);
		if (strcmp(section->header, "[data]") == 0) print_cfg_data((cfg_data*)section);
		else if (strcmp(section->header, "[net]") == 0) print_cfg_net((cfg_net*)section);
		else if (strcmp(section->header, "[training]") == 0) print_cfg_training((cfg_training*)section);
		else if (strcmp(section->header, "[conv]") == 0) print_cfg_conv((cfg_conv*)section);
		else if (strcmp(section->header, "[classify]") == 0) print_cfg_classify((cfg_classify*)section);
	}
	printf("[END CFG]\n\n");
}

void print_cfg_data(cfg_data* s) {
	printf("\n[data]\n");
	printf("dataset_dir = %s\n", s->dataset_dir);
	printf("classes_file = %s\n", s->classes_file);
	printf("weights_file = %s\n", s->weights_file);
	printf("backup_dir = %s\n", s->backup_dir);
}

void print_cfg_net(cfg_net* s) {
	printf("\n[net]\n");
	printf("width = %zu\nheight = %zu\nchannels = %zu\n", s->width, s->height, s->channels);
	printf("num_classes = %zu\n", s->num_classes);
	printf("cost = ");
	print_cost_type(s->cost);
}

void print_cfg_training(cfg_training* s) {
	printf("\n[training]\n");
	printf("batch_size = %zu\n", s->batch_size);
	printf("subbatch_size = %zu\n", s->subbatch_size);
	printf("max_iterations = %zu\n", s->max_iterations);
	printf("learning_rate = %f\n", s->learning_rate);
	if (s->lr_policy == LR_STEPS) {
		printf("lr_policy = steps\n");
		printf("step_percents = ");
		print_floatarr(&(s->step_percents));
		printf("step_scaling = ");
		print_floatarr(&(s->step_scaling));
	}
	printf("ease_in = %zu\n", s->ease_in);
	printf("momentum = %f\n", s->momentum);
	printf("decay = %f\n", s->decay);
	printf("saturation = ");
	print_floatarr(&(s->saturation));
	printf("exposure = ");
	print_floatarr(&(s->exposure));
	printf("hue = ");
	print_floatarr(&(s->hue));
}

void print_cfg_conv(cfg_conv* s) {
	printf("\n[conv]\n");
	printf("id = %d\n", s->id);
	printf("train = %d\n", s->train);
	printf("batch_normalize = %d\n", s->batch_normalize);
	printf("filters = %zu\n", s->n_filters);
	printf("kernel_size = %zu\n", s->kernel_size);
	printf("stride = %zu\n", s->stride);
	printf("pad = %zu\n", s->pad);
	printf("activation = ");
	print_activation(s->activation);
	printf("in_ids = ");
	print_intarr(&(s->in_ids));
	printf("out_ids = ");
	print_intarr(&(s->out_ids));
}

void print_cfg_classify(cfg_classify* s) {
	printf("\n[classify]\n");
	printf("id = %d\n", s->id);
	printf("train = %d\n", s->train);
	printf("cost = ");
	print_cost_type(s->cost);
	printf("in_ids = ");
	print_intarr(&(s->in_ids));
}
