#include "config.h"
#include "xallocs.h"
#include "list.h"
#include "file_utils.h"
#include "utils.h"
#include <stdio.h>
#include <string.h>



char* read_line(FILE* file);
void clean_string(char* s);
char** split_string(char* str, const char* delimiters);
int char_in_string(char c, char* str);
void print_tokens(char** tokens);
void append_cfg_section(list* lst, char* header);

static char* headers[] = { "[input]", "[training]", "[conv]", "[yolo]", "\0"};



void load_cfg(char* filename, network* net) {
	FILE* file = get_file(filename, "r");
	char* line;
	list* sections = new_list();
	cfg_section* section;
	int h = -1;  // index of headers[] corresponding to current header being read
	while (line = read_line(file), line != 0) {
		clean_string(line);
		char** tokens = split_string(line, "=,");
		if (tokens[0][0] == '[') {
			int i = 0;
			while (headers[i][0] != '\0') {
				if (strcmp(tokens[0], headers[i]) == 0) {
					h = i;
					append_cfg_section(sections, headers[h]);
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
	}
	print_cfg(sections);
}

void append_cfg_section(list* lst, char* header) {
	if (strcmp(header, headers[0])) {
		list_append(lst, new_cfg_input());
		return 0;
	}
	if (strcmp(header, headers[1])) {
		list_append(lst, new_cfg_training());
		return 0;
	}
	if (strcmp(header, headers[2])) {
		list_append(lst, new_cfg_conv());
		return 0;
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
	section->set_param = set_param_cfg_input;
	return section;
}

cfg_conv* new_cfg_conv() {
	cfg_conv* section = (cfg_conv*)xcalloc(1, sizeof(cfg_conv));
	section->header = headers[2];
	section->set_param = set_param_cfg_conv;
	return section;
}

void set_param_cfg_input(void* section, char** tokens) {
	cfg_input* sec = (cfg_input*)section;
	int ret;
	char* param = tokens[0];  //pointer to name of param
	if (strcmp(tokens[0], "width") == 0) {
		ret = str2int(tokens[1]);
		sec->width = ret;
		return 0;
	}
	if (strcmp(param, "height") == 0) {
		ret = str2int(tokens[1]);
		sec->height = ret;
		return 0;
	}
	if (strcmp(param, "channels") == 0) {
		ret = str2int(tokens[1]);
		sec->channels = ret;
		return 0;
	}
	printf("Error: No parameter named %s in section %s.", param, sec->header);
}

void set_param_cfg_training(void* section, char** tokens) {
	cfg_training* sec = (cfg_training*)section;

}

void set_param_cfg_conv(void* section, char** tokens) {
	cfg_conv* sec = (cfg_conv*)section;

}



/*
Allocates char array of size 512 and stores result of fgets.
Returns array on success.
Returns 0 if fgets failed.
*/
char* read_line(FILE* file) {
	size_t size = 512;
	char* line = (char*)xmalloc(size * sizeof(char));
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
char** split_string(char* str, const char* delimiters) {
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

int char_in_string(char c, char* str) {
	size_t length = strlen(str);
	size_t i;
	for (i = 0; i < length; i++) {
		if (c == str[i]) return 1;
	}
	return 0;
}

void print_tokens(char** tokens) {
	size_t i = 0;
	while (tokens[i] != NULL) {
		printf("token %zu = %s\n", i, tokens[i]);
		i++;
	}
}

void print_cfg(list* sections) {
	size_t n = sections->size;
	cfg_section* section;
	for (int i = 0; i < n; i++) {
		section = (cfg_section*)list_get_item(sections, i);
		if (strcmp(section->header, "[input]") == 0) {
			cfg_input* s = (cfg_input*)section;
			printf("width = %d\nheight = %d\nchannels = %d", s->width, s->height, s->channels);
		}
	}
}