#include "cfg2.h"
#include "xallocs.h"
#include "utils.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>


#define hDATA "[data]"
#define hNET "[net]"
#define hTRAINING "[training]"
#define hCONV "[conv]"
#define hCLASSIFY "[classify]"
#define hDETECT "[detect]"

#define LINESIZE 512
#define TOKENSSIZE 64


int is_header(char* str, int* is_layer, char** header);


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
	cfg c = { 0 };
	char line[LINESIZE] = { 0 };
	char* tokens[TOKENSSIZE] = { 0 };
	char* header[100] = { 0 };
	int is_layer = 0;
	int ok;
	while (ok = read_line_to_buff(file, line, LINESIZE), ok) {
		clean_string(line);
		split_string_to_buff(line, "=,", tokens);
		if (tokens[0] == NULL) continue;
		if (is_header(line, &is_layer, &header)) {
			if (is_layer) (*n_layers)++;
			copy_to_cfg(&c, tokens);
		}
		else {
			section = (cfg_section*)(sections->last->val);
			section->set_param(section, tokens);
		}
	}
	close_filestream(file);
	//print_cfg(sections);
	return sections;
}

void copy_to_cfg(cfg* c, char** tokens) {
	int t = tokens_length(tokens);
	if (t < 2) {
		printf("Error. Tokens must have a minimum length of 2, has %d.\n", t);
		wait_for_key_then_exit();
	}
	char* k = tokens[0];
	if (strcmp(k, "dataset_dir") == 0) {
		c->dataset_dir = (char*)xcalloc(strlen(tokens[1] + 1), sizeof(char));
		strcpy(c->dataset_dir, tokens[1]);
	}
	else if (strcmp(k, "classes_file") == 0) {
		c->classes_file = (char*)xcalloc(strlen(tokens[1] + 1), sizeof(char));
		strcpy(c->classes_file, tokens[1]);
	}
	else if (strcmp(k, "weights_file") == 0) {
		c->weights_file = (char*)xcalloc(strlen(tokens[1] + 1), sizeof(char));
		strcpy(c->weights_file, tokens[1]);
	}
	else if (strcmp(k, "backup_dir") == 0) {
		c->backup_dir = (char*)xcalloc(strlen(tokens[1] + 1), sizeof(char));
		strcpy(c->backup_dir, tokens[1]);
	}
	else if (strcmp(k, "width") == 0) {
		c->width = str2sizet(tokens[1]);
	}
}

int tokens_length(char** tokens) {
	int i = 0;
	while (tokens[i] != NULL) i++;
	return i;
}

int is_header(char* str, int* is_layer, char** header) {
	*is_layer = 0;
	*header = 0;
	if (str[0] != '[') return 0;
	if (strcmp(str, hDATA) == 0) {
		*header = &hDATA;
		return 1;
	}
	else if (strcmp(str, hNET) == 0) {
		*header = &hNET;
		return 1;
	}
	else if (strcmp(str, hTRAINING) == 0) {
		*header = &hTRAINING;
		return 1;
	}
	else if (strcmp(str, hCONV) == 0) {
		*header = &hCONV;
		*is_layer = 1;
		return 1;
	}
	else if (strcmp(str, hCLASSIFY) == 0) {
		*header = &hCLASSIFY;
		*is_layer = 1;
		return 1;
	}
	else if (strcmp(str, hDETECT) == 0) {
		*header = &hCLASSIFY;
		*is_layer = 1;
		return 1;
	}
	return 0;
}