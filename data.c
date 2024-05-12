#include "data.h"
#include "utils.h"
#include "xallocs.h"


void load_annotations(char* filename) {
	FILE* file = get_filestream(filename, "r");
	size_t n = get_line_count(filename);
	float* annots = (float*)xcalloc(n * 5, sizeof(float));
	while (line = read_line(file), line != 0) {
		clean_string(line);
		char** tokens = split_string(line, ",");
		if (tokens == NULL) {
			xfree(line);
			continue;
		}
	}
}
