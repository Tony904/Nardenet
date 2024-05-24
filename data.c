#include "data.h"
#include <assert.h>
#include <string.h>
#include "utils.h"
#include "xallocs.h"
#include "image.h"

// MOVE THIS FUNCTIONALITY TO CFG.C
data_paths* get_data_paths(char* datafile) {
	if (!file_exists(datafile)) wait_for_key_then_exit();
	FILE* file = get_filestream(datafile, "r");
	data_paths* dp = (data_paths*)xcalloc(1, sizeof(data_paths));
	size_t n = 0;
	char* line;
	while (line = read_line(file), line != 0) {
		clean_string(line);
		char** tokens = split_string(line, "=");
		if (tokens == NULL) {
			printf("Error parsing string.\n");
			printf("Line %zu, file %s\n", n, datafile);
			wait_for_key_then_exit();
		}
		if (strcmp(tokens[0], "data_dir") == 0) {
			dp->data_dir = tokens[1];
		}
		else if (strcmp(tokens[0], "images_dir") == 0) {
			dp->imgs_dir = tokens[1];
		}
		else if (strcmp(tokens[0], "annotations_dir") == 0) {
			dp->ants_dir = tokens[1];
		}
		else if (strcmp(tokens[0], "classes_file") == 0) {
			dp->classes_file = tokens[1];
		}
		else if (strcmp(tokens[0], "backup_dir") == 0) {
			dp->backup_dir = tokens[1];
		}
		else if (strcmp(tokens[0], "weights_file") == 0) {
			dp->weights_file = tokens[1];
		}
		else {
			printf("Unrecognized key %s\n", tokens[0]);
		}
	}
	return dp;
}

void free_data_paths(data_paths_od* dp) {
	xfree(dp->ants_dir);
	xfree(dp->backup_dir);
	xfree(dp->classes_file);
	xfree(dp->data_dir);
	xfree(dp->imgs_dir);
	xfree(dp->weights_file);
	xfree(dp);
}