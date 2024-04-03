#include "cfg.h"
#include "list.h"
#include "xallocs.h"
#include "file_utils.h"
#include "utils.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>


list* get_cfg_sections(char* filename, size_t* n_layers);
char* read_line(FILE* file);
void clean_string(char* s);
char** split_string(char* str, char* delimiters);
void append_cfg_section(list* lst, char* header);
cfg_input* new_cfg_input(void);
cfg_training* new_cfg_training(void);
cfg_conv* new_cfg_conv(void);
void set_param_cfg_input(cfg_section* section, char** tokens);
void set_param_cfg_training(cfg_section* section, char** tokens);
void set_param_cfg_conv(cfg_section* section, char** tokens);
int* tokens2intarray(char** tokens, size_t offset, size_t* array_length);
LR_POLICY str2lr_policy(char* str);
ACTIVATION str2activation(char* str);
floatarr tokens2floatarr(char** tokens, size_t offset);
intarr tokens2intarr(char** tokens, size_t offset);
size_t tokens_length(char** tokens);
void print_tokens(char** tokens);
void print_cfg_input(cfg_input* s);
void print_cfg_training(cfg_training* s);
void print_cfg_conv(cfg_conv* s);
void print_cfg(list* sections);
void print_tokens(char** tokens);
LAYER_TYPE header2layertype(char* header);
int is_layer_header(char* header);
void load_cfg_to_network(list* sections, network* net);
void load_cfg_input_to_network(cfg_input* s, network* n);
void load_cfg_training_to_network(cfg_training* s, network* n);
int load_cfg_section_to_layer(cfg_section* s, network* n);
void load_cfg_conv_to_layer(cfg_conv* s, layer* l);
void load_cfg_yolo_to_layer(cfg_yolo* s, layer* l);
void free_sections(list* lst);
void free_section(cfg_section* s);
void free_cfg_input(cfg_input* s);
void free_cfg_training(cfg_training* s);
void free_cfg_conv(cfg_conv* s);
void free_cfg_yolo(cfg_yolo* s);

static char* headers[] = { "[input]", "[training]", "[conv]", "[yolo]", "\0"};


network* create_network_from_cfg(char* cfg_filename) {
	size_t n_layers = 0;
	list* sections = get_cfg_sections(cfg_filename, &n_layers);
	network* net = new_network(n_layers);
	load_cfg_to_network(sections, net);
	free_sections(sections);
	build_network(net);
	return net;
}

void load_cfg_to_network(list* sections, network* net) {
	size_t n = net->n_layers;
	node* noed = sections->first;
	cfg_section* sec;
	size_t i = 0;
	while (noed != NULL) {
		sec = (cfg_section*)noed->val;
		if (strcmp(sec->header, "[input]") == 0) {
			load_cfg_input_to_network((cfg_input*)sec, net);
		}
		else if (strcmp(sec->header, "[training]") == 0) {
			load_cfg_training_to_network((cfg_training*)sec, net);
		}
		else if (load_cfg_section_to_layer(sec, net)) {
			i++;
		}
		noed = noed->next;
	}
	assert(n == i); // number of layers counted equals number of layers loaded
}

void load_cfg_input_to_network(cfg_input* s, network* n) {
	n->w = s->width;
	n->h = s->height;
	n->c = s->channels;
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
	if (ltype == CONV) {
		load_cfg_conv_to_layer((cfg_conv*)sec, &(net->layers[id]));
		return 1;
	}
	if (ltype == YOLO) {
		load_cfg_yolo_to_layer((cfg_yolo*)sec, &(net->layers[id]));
		return 1;
	}
	return 0;
}

void load_cfg_conv_to_layer(cfg_conv* s, layer* l) {
	assert(l->type == (LAYER_TYPE)0); // check layer not already written to
	l->type = CONV;
	l->id = s->id;
	l->batch_norm = s->batch_normalize;
	l->n_filters = s->n_filters;
	l->ksize = s->kernel_size;
	l->stride = s->stride;
	l->pad = s->pad;
	l->in_ids = s->in_ids;
	l->out_ids = s->out_ids;
}

void load_cfg_yolo_to_layer(cfg_yolo* s, layer* l) {
	l->type = CONV;
	l->id = s->id;
	l->n_filters = s->n_filters;
	l->ksize = s->kernel_size;
	l->stride = s->stride;
	l->pad = s->pad;
	l->anchors = s->anchors;
	l->in_ids = s->in_ids;
}

void append_cfg_section(list* lst, char* header) {
	if (strcmp(header, headers[0]) == 0) {
		list_append(lst, new_cfg_input());
		return;
	}
	if (strcmp(header, headers[1]) == 0) {
		list_append(lst, new_cfg_training());
		return;
	}
	if (strcmp(header, headers[2]) == 0) {
		list_append(lst, new_cfg_conv());
		return;
	}
}

cfg_input* new_cfg_input(void) {
	cfg_input* section = (cfg_input*)xcalloc(1, sizeof(cfg_input));
	section->header = headers[0];
	section->set_param = set_param_cfg_input;
	return section;
}

cfg_training* new_cfg_training(void) {
	cfg_training* section = (cfg_training*)xcalloc(1, sizeof(cfg_training));
	section->header = headers[1];
	section->set_param = set_param_cfg_training;
	return section;
}

cfg_conv* new_cfg_conv(void) {
	cfg_conv* section = (cfg_conv*)xcalloc(1, sizeof(cfg_conv));
	section->header = headers[2];
	section->set_param = set_param_cfg_conv;
	// Set non-zero defaults
	section->train = 1;
	return section;
}

void set_param_cfg_input(cfg_section* section, char** tokens) {
	cfg_input* sec = (cfg_input*)section;
	char* param = tokens[0];  //pointer to name of param
	if (strcmp(param, "width") == 0) {
		sec->width = str2sizet(tokens[1]);
		return;
	}
	if (strcmp(param, "height") == 0) {
		sec->height = str2sizet(tokens[1]);
		return;
	}
	if (strcmp(param, "channels") == 0) {
		sec->channels = str2sizet(tokens[1]);
		return;
	}
	fprintf(stderr, "Error: No parameter named %s in section %s.\n", param, sec->header);
	exit(EXIT_FAILURE);
}

void set_param_cfg_training(cfg_section* section, char** tokens) {
	cfg_training* sec = (cfg_training*)section;
	char* param = tokens[0];
	if (strcmp(param, "batch_size") == 0) {
		sec->batch_size = str2sizet(tokens[1]);
		return;
	}
	if (strcmp(param, "subbatch_size") == 0) {
		sec->subbatch_size = str2sizet(tokens[1]);
		return;
	}
	if (strcmp(param, "max_iterations") == 0) {
		sec->max_iterations = str2sizet(tokens[1]);
		return;
	}
	if (strcmp(param, "learning_rate") == 0) {
		sec->learning_rate = str2float(tokens[1]);
		return;
	}
	if (strcmp(param, "lr_policy") == 0) {
		sec->lr_policy = str2lr_policy(tokens[1]);
		return;
	}
	if (strcmp(param, "step_percents") == 0) {
		sec->step_percents = tokens2floatarr(tokens, 1);
		return;
	}
	if (strcmp(param, "step_scaling") == 0) {
		sec->step_scaling = tokens2floatarr(tokens, 1);
		return;
	}
	if (strcmp(param, "ease_in") == 0) {
		sec->ease_in = str2sizet(tokens[1]);
		return;
	}
	if (strcmp(param, "momentum") == 0) {
		sec->momentum = str2float(tokens[1]);
		return;
	}
	if (strcmp(param, "decay") == 0) {
		sec->decay = str2float(tokens[1]);
		return;
	}
	if (strcmp(param, "saturation") == 0) {
		sec->saturation = tokens2floatarr(tokens, 1);
		return;
	}
	if (strcmp(param, "exposure") == 0) {
		sec->exposure = tokens2floatarr(tokens, 1);
		return;
	}
	if (strcmp(param, "hue") == 0) {
		sec->hue = tokens2floatarr(tokens, 1);
		return;
	}
}

void set_param_cfg_conv(cfg_section* section, char** tokens) {
	cfg_conv* sec = (cfg_conv*)section;
	char* param = tokens[0];
	if (strcmp(param, "id") == 0) {
		sec->id = str2int(tokens[1]);
		return;
	}
	if (strcmp(param, "batch_normalize") == 0) {
		sec->batch_normalize = str2int(tokens[1]);
		return;
	}
	if (strcmp(param, "train") == 0) {
		sec->train = str2int(tokens[1]);
		return;
	}
	if (strcmp(param, "filters") == 0) {
		sec->n_filters = str2sizet(tokens[1]);
		return;
	}
	if (strcmp(param, "kernel_size") == 0) {
		sec->kernel_size = str2sizet(tokens[1]);
		return;
	}
	if (strcmp(param, "stride") == 0) {
		sec->stride = str2sizet(tokens[1]);
		return;
	}
	if (strcmp(param, "pad") == 0) {
		sec->pad = str2sizet(tokens[1]);
		return;
	}
	if (strcmp(param, "activation") == 0) {
		sec->activation = str2activation(tokens[1]);
		return;
	}
	if (strcmp(param, "in_ids") == 0) {
		sec->in_ids = tokens2intarr(tokens, 1);
		return;
	}
	if (strcmp(param, "out_ids") == 0) {
		sec->out_ids = tokens2intarr(tokens, 1);
		return;
	}
}

list* get_cfg_sections(char* filename, size_t* n_layers) {
	*n_layers = 0;
	FILE* file = get_file(filename, "r");
	char* line;
	list* sections = new_list();
	cfg_section* section;
	int h = -1;  // index of headers[] corresponding to current header being read
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

void free_sections(list* l) {
	node* n = l->first;
	while (n) {
		free_section((cfg_section*)n->val);
		n = n->next;
	}
	free_list(l);
}

void free_section(cfg_section* sec) {
	if (strcmp(sec->header, headers[0]) == 0) {
		free_cfg_input((cfg_input*)sec);
		return;
	}
	if (strcmp(sec->header, headers[1]) == 0) {
		free_cfg_training((cfg_training*)sec);
		return;
	}
	if (strcmp(sec->header, headers[2]) == 0) {
		free_cfg_conv((cfg_conv*)sec);
		return;
	}
	if (strcmp(sec->header, headers[3]) == 0) {
		free_cfg_yolo((cfg_yolo*)sec);
		return;
	}
}

void free_cfg_input(cfg_input* s) {
	xfree(s);
}

void free_cfg_training(cfg_training* s) {
	xfree(s->step_percents.a);
	xfree(s->step_scaling.a);
	xfree(s->saturation.a);
	xfree(s->exposure.a);
	xfree(s->hue.a);
	xfree(s);
}

void free_cfg_conv(cfg_conv* s) {
	//xfree(s->in_ids.a);  keep around to be stored in layer
	//xfree(s->out_ids.a);
	xfree(s);
}

void free_cfg_yolo(cfg_yolo* s) {
	//xfree(s->in_ids.a);
	xfree(s->anchors);
	xfree(s);
}

int is_layer_header(char* header) {
	if (strcmp(header, "[conv]") == 0) return 1;
	if (strcmp(header, "[yolo]") == 0) return 1;
	else return 0;
}

LAYER_TYPE header2layertype(char* header) {
	if (strcmp(header, "[conv]") == 0) return CONV;
	if (strcmp(header, "[yolo]") == 0) return YOLO;
	return NONE_LAYER;
}

/*
Allocates char array of size 512 and stores result of fgets.
Returns array on success.
Returns 0 if fgets failed.
*/
char* read_line(FILE* file) {
	int size = 512;
	char* line = (char*)xcalloc(size, sizeof(char));
	if (!fgets(line, size, file)) {  // fgets returns null pointer on fail or end-of-file
		xfree(line);
		return 0;
	}
	return line;
}

/*
Removes whitespaces and line-end characters.
Removes comment character '#' and all characters after.
*/
void clean_string(char* str) {
	size_t length = strlen(str);
	size_t offset = 0;
	size_t i;
	char c;
	for (i = 0; i < length; i++) {
		c = str[i];
		if (c == '#') break;  // '#' is used for comments
		if (c == ' ' || c == '\n' || c == '\r') offset++;
		else str[i - offset] = c;
	}
	str[i - offset] = '\0';
}

/*
Splits string by delimiter and returns a null-terminated char* array with pointers to str.
Modifies str.
*/
char** split_string(char* str, char* delimiters) {
	size_t length = strlen(str);
	if (!length) return NULL;
	size_t i = 0;
	if (char_in_string(str[0], delimiters) || char_in_string(str[length - 1], delimiters)) {
		fprintf(stderr, "Line must not start or end with delimiter.\nDelimiters: %s\n Line: %s\n", delimiters, str);
		exit(EXIT_FAILURE);
	}
	size_t count = 1;
	for (i = 0; i < length; i++) {
		if (char_in_string(str[i], delimiters)) {
			if (char_in_string(str[i + 1], delimiters)) {
				fprintf(stderr, "Line must not contain consecutive delimiters.\nDelimiters: %s\n Line: %s\n", delimiters, str);
				exit(EXIT_FAILURE);
			}
			count++;
		}
	}
	char** strings = (char**)xcalloc(count + 1, sizeof(char*));
	strings[0] = &str[0];
	size_t j = 1;
	if (count > 1)
		for (i = 1; i < length; i++) {
			if (char_in_string(str[i], delimiters)) {
				str[i] = '\0';
				strings[j] = &str[i + 1];
				j++;
				i++;
			}
		}
	strings[count] = NULL;
	return strings;
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

size_t tokens_length(char** tokens) {
	size_t i = 0;
	while (tokens[i] != NULL) {
		i++;
	}
	return i;
}

LR_POLICY str2lr_policy(char* str) {
	if (strcmp(str, "steps") == 0) {
		return LR_STEPS;
	}
	fprintf(stderr, "Error: No valid policy named %s.\n", str);
	exit(EXIT_FAILURE);
}

ACTIVATION str2activation(char* str) {
	if (strcmp(str, "relu") == 0) {
		return RELU;
	}
	if (strcmp(str, "mish") == 0) {
		return MISH;
	}
	fprintf(stderr, "Error: No valid activation named %s.\n", str);
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
		if (strcmp(section->header, "[input]") == 0) {
			print_cfg_input((cfg_input*)section);
			continue;
		}
		if (strcmp(section->header, "[training]") == 0) {
			print_cfg_training((cfg_training*)section);
			continue;
		}
		if (strcmp(section->header, "[conv]") == 0) {
			print_cfg_conv((cfg_conv*)section);
			continue;
		}
	}
	printf("[END CFG]\n\n");
}

void print_cfg_input(cfg_input* s) {
	printf("\n[SECTION]\n");
	printf("[input]\n");
	printf("width = %zu\nheight = %zu\nchannels = %zu\nSECTION END\n", s->width, s->height, s->channels);
	printf("[END SECTION]\n");
}

void print_cfg_training(cfg_training* s) {
	printf("\n[SECTION]\n");
	printf("[training]\n");
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
	printf("[END SECTION]\n");
}

void print_cfg_conv(cfg_conv* s) {
	printf("\n[SECTION]\n");
	printf("[conv]\n");
	printf("id = %d\n", s->id);
	printf("train = %d\n", s->train);
	printf("batch_normalize = %d\n", s->batch_normalize);
	printf("filters = %zu\n", s->n_filters);
	printf("kernel_size = %zu\n", s->kernel_size);
	printf("stride = %zu\n", s->stride);
	printf("pad = %zu\n", s->pad);
	if (s->activation == RELU) {
		printf("activation = relu\n");
	}
	printf("in_ids = ");
	print_intarr(&(s->in_ids));
	printf("out_ids = ");
	print_intarr(&(s->out_ids));
	printf("[END SECTION]\n");
}
