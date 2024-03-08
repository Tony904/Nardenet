#include "cfg_reader.h"
#include "nardenet.h"
#include "xallocs.h"
#include "file_utils.h"
#include <stdio.h>
#include <string.h>


char* read_line(FILE* file);
void clean_string(char* s);


void load_cfg(char* filename, network net) {
	FILE* file = get_file(filename, "r");
	char* line;
	while (line = read_line(file), line != 0) {
		clean_string(line);
	}
	

}

char* read_line(FILE* file) {
	size_t size = 512;
	char* line = (char*)xmalloc(size * sizeof(char));
	if (!fgets(line, size, file)) {
		free(line);
		return 0;
	}
	return line;
}

void clean_string(char* s) {
	size_t length = strlen(s);
	size_t offset = 0;
	size_t i;
	char c;
	for (i = 0; i < length; i++) {
		c = s[i];
		if (c == '#') break;  // '#' is used for comments
		if (c == ' ' || c == '\n' || c == '\r') offset++;
		else s[i - offset] = c;
	}
	s[i - offset] = '\0';
}
