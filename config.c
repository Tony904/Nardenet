#include "config.h"
#include "nardenet.h"
#include "network.c"
#include "xallocs.h"
#include "list.h"
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
cfg_input* new_cfg_input();
cfg_training* new_cfg_training();
cfg_conv* new_cfg_conv();
void set_param_cfg_input(cfg_section* section, char** tokens);
void set_param_cfg_training(cfg_section* section, char** tokens);
void set_param_cfg_conv(cfg_section* section, char** tokens);
int* tokens2intarray(char** tokens, size_t offset, size_t* array_length);
LR_POLICY str2lr_policy(char* str);
ACTIVATION str2activation(char* str);
floatarr* tokens2floatarr(char** tokens, size_t offset);
size_t tokens_length(char** tokens);
void print_tokens(char** tokens);
void print_cfg_input(cfg_input* s);
void print_cfg_training(cfg_training* s);
void print_cfg_conv(cfg_conv* s);
void print_cfg(list* sections);
void print_tokens(char** tokens);
LAYER_TYPE header2layertype(char* header);
int load_cfg_section_to_layer(cfg_section* sec, layer* lay);
void load_cfg_conv_to_layer(cfg_conv* section, layer* lay);

static char* headers[] = { "[input]", "[training]", "[conv]", "[yolo]", "\0"};



network* create_network_from_cfg(char* cfg_filename) {
	size_t n_layers = 0;
	list* sections = get_cfg_sections(cfg_filename, &n_layers);
	network* net = new_network(n_layers);
	load_layers_to_network(net, sections);

}

void load_layers_to_network(network* net, list* sections) {
	size_t length = sections->length;
	size_t n = net->n_layers;
	node* noed = sections->first;
	cfg_section* sec;
	size_t i = 0;
	while (noed != NULL) {
		sec = (cfg_section*)noed->val;
		if (sec->header == "[input]") {
			load_cfg_input_to_network((cfg_input*)sec, net);
		}
		else if (sec->header == "[training]") {
			load_cfg_training_to_network((cfg_training*)sec, net);
		}
		else if (load_cfg_section_to_layer(sec, &(net->layers[i]))) {
			i++;
		}
		noed = noed->next;
	}
	assert(n == i); // number of layers counted equals number of layers loaded
}

void load_cfg_input_to_network(cfg_input* sec, network* net) {
	net->w = sec->width;
	net->h = sec->height;
	net->c = sec->channels;
}

void load_cfg_training_to_network(cfg_training* s, network* n) {
	n->batch_size = s->batch_size;
	n->subbatch_size = s->subbatch_size;
	n->max_iterations = s->max_iterations;
	n->learning_rate = s->learning_rate;
	n->lr_policy = s->lr_policy;
	n->step_percents = *(s->step_percents);
	n->step_scaling = *(s->step_scaling);
	n->ease_in = s->ease_in;
	n->momentum = s->momentum;
	n->decay = s->decay;
	n->saturation[0] = 1.0;
	n->saturation[1] = 1.0;
	if (s->saturation->length > 0) {
		n->saturation[0] = s->saturation->vals[0];
		if (s->saturation->length == 2) {
			n->saturation[1] = s->saturation->vals[1];
		}
	}
	n->exposure[0] = 1.0;
	n->exposure[1] = 1.0;
	if (s->exposure->length > 0) {
		n->exposure[0] = s->saturation->vals[0];
		if (s->saturation->length == 2) {
			n->saturation[1] = s->saturation->vals[1];
		}
	}
	n->hue[0] = 1.0;
	n->hue[1] = 1.0;
	if (s->hue->length > 0) {
		n->hue[0] = s->hue->vals[0];
		if (s->hue->length == 2) {
			n->hue[1] = s->hue->vals[1];
		}
	}
}

int load_cfg_section_to_layer(cfg_section* sec, layer* lay) {
	LAYER_TYPE ltype = header2layertype(sec->header);
	if (ltype == CONV) {
		load_cfg_conv_to_layer((cfg_conv*)sec, lay);
		return 1;
	}
	if (ltype == YOLO) {
		load_cfg_yolo_to_layer((cfg_yolo*)sec, lay);
		return 1;
	}
	return 0;
}

void load_cfg_conv_to_layer(cfg_conv* s, layer* l) {
	l->type = CONV;
	l->id = s->id;
	l->batch_norm = s->batch_normalize;
	l->n_filters = s->n_filters;
	l->k_size = s->kernel_size;
	l->stride = s->stride;
	l->padding = s->pad;
}

void load_cfg_yolo_to_layer(cfg_yolo* s, layer* l) {
	l->type = CONV;
	l->id = s->id;
	l->n_filters = s->n_filters;
	l->k_size = s->kernel_size;
	l->stride = s->stride;
	l->padding = s->pad;
	l->anchors = s->anchors;
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
		if (tokens == NULL) continue;
		if (tokens[0][0] == '[') {
			int i = 0;
			while (headers[i][0] != '\0') {
				if (strcmp(tokens[0], headers[i]) == 0) {
					h = i;
					printf("Appending cfg section: %s\n", headers[h]);
					append_cfg_section(sections, headers[h]);
					if (is_layer_header(headers[h])) *n_layers++;
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
		xfree(tokens);
	}
	close_filestream(file);
	print_cfg(sections);
	return sections;
}

int is_layer_header(char* header) {
	if (strcmp(header, "[conv]") == 0) return 1;
	if (strcmp(header, "[yolo]") == 0) return 1;
	else return 0;
}

LAYER_TYPE header2layertype(char* header) {
	if (strcmp(header, "[conv]") == 0) return CONV;
	if (strcmp(header, "[yolo]") == 0) return YOLO;
	return NOT_A_LAYER;
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

cfg_input* new_cfg_input() {
	cfg_input* section = (cfg_input*)xcalloc(1, sizeof(cfg_input));
	section->header = headers[0];
	section->set_param = set_param_cfg_input;
	return section;
}

cfg_training* new_cfg_training() {
	cfg_training* section = (cfg_training*)xcalloc(1, sizeof(cfg_training));
	section->header = headers[1];
	section->set_param = set_param_cfg_training;
	return section;
}

cfg_conv* new_cfg_conv() {
	cfg_conv* section = (cfg_conv*)xcalloc(1, sizeof(cfg_conv));
	section->header = headers[2];
	section->set_param = set_param_cfg_conv;
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
		sec->id = str2sizet(tokens[1]);
		return;
	}
	if (strcmp(param, "batch_normalize") == 0) {
		sec->batch_normalize = str2sizet(tokens[1]);
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
		free(line);
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

floatarr* tokens2floatarr(char** tokens, size_t offset) {
	size_t n = tokens_length(tokens);
	assert(n > offset);
	floatarr* farr = (floatarr*)xcalloc(1, sizeof(floatarr));
	size_t length = n - offset;
	farr->length = length;
	farr->vals = (float*)xcalloc(length, sizeof(float));
	for (size_t i = 0; i < length; i++) {
		farr->vals[i] = str2float(tokens[i + offset]);
	}
	return farr;
}

int* tokens2intarray(char** tokens, size_t offset, size_t* array_length) {
	size_t n = tokens_length(tokens);
	assert(n > offset);
	size_t length = n - offset;
	*array_length = length;
	int* iarray = (int*)xcalloc(length, sizeof(int));
	for (size_t i = 0; i < length; i++) {
		iarray[i] = str2int(tokens[i + offset]);
	}
	return iarray;
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

void print_floatarr(floatarr* p) {
	size_t n = p->length;
	assert(n > 0);
	size_t i;
	for (i = 0; i < n - 1; i++) {
		printf("%f, ", p->vals[i]);
	}
	printf("%f\n", p->vals[i]);
}

void print_cfg(list* sections) {
	printf("\nprint_cfg():\n");
	size_t n = sections->length;
	printf("# of sections = %zu\n\n", n);
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
	printf("END CFG\n\n");
}

void print_cfg_input(cfg_input* s) {
	printf("SECTION: [input]\n");
	printf("width = %zu\nheight = %zu\nchannels = %zu\nSECTION END\n\n", s->width, s->height, s->channels);
}

void print_cfg_training(cfg_training* s) {
	printf("SECTION: [training]\n");
	printf("batch_size = %zu\n", s->batch_size);
	printf("subbatch_size = %zu\n", s->subbatch_size);
	printf("max_iterations = %zu\n", s->max_iterations);
	printf("learning_rate = %f\n", s->learning_rate);
	if (s->lr_policy == LR_STEPS) {
		printf("lr_policy = steps\n");
		printf("step_percents = ");
		print_floatarr(s->step_percents);
		printf("step_scaling = ");
		print_floatarr(s->step_scaling);
	}
	printf("ease_in = %zu\n", s->ease_in);
	printf("momentum = %f\n", s->momentum);
	printf("decay = %f\n", s->decay);
	printf("saturation = ");
	print_floatarr(s->saturation);
	printf("exposure = ");
	print_floatarr(s->exposure);
	printf("hue = ");
	print_floatarr(s->hue);
	printf("SECTION END\n\n");
}

void print_cfg_conv(cfg_conv* s) {
	printf("SECTION: [conv]\n");
	printf("batch_normalize = %zu\n", s->batch_normalize);
	printf("filters = %zu\n", s->n_filters);
	printf("kernel_size = %zu\n", s->kernel_size);
	printf("stride = %zu\n", s->stride);
	printf("pad = %zu\n", s->pad);
	if (s->activation == RELU) {
		printf("activation = relu\n");
	}
	printf("SECTION END\n\n");
}
