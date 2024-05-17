#include "data.h"
#include <assert.h>
#include <string.h>
#include "utils.h"
#include "xallocs.h"
#include "image.h"


sample* load_samples(char* folder) {

}

sample* load_sample(char* antfile, char* imgfile) {
	if (!file_exists(imgfile)) wait_for_key_then_exit();
	FILE* file = get_filestream(antfile, "r");
	size_t n_lines = get_line_count(file);
	sample* samp = (sample*)xcalloc(1, sizeof(sample));
	samp->nboxes = n_lines;
	samp->imgpath = imgfile;
	samp->bboxes = (bbox*)xcalloc(n_lines, sizeof(bbox));
	char* line;
	size_t n = 0;
	while (line = read_line(file), line != 0) {
		clean_string(line);
		char** tokens = split_string(line, ",");
		if (tokens == NULL) {
			printf("Error parsing bounding box string.\n");
			printf("Line %zu, file %s\n", n, antfile);
			wait_for_key_then_exit();
		}
		size_t length = tokens_length(tokens);
		if (length != 5) {
			printf("Bounding boxes must have 5 parameters. Has %zu.\n", length);
			printf("Line %zu, file %s\n", n, antfile);
			wait_for_key_then_exit();
		}
		bbox* box = &samp->bboxes[n];
		box->lbl = str2int(tokens[0]);
		box->cx = str2float(tokens[1]);
		box->cy = str2float(tokens[2]);
		box->w = str2float(tokens[3]);
		box->h = str2float(tokens[4]);
		n++;
	}
	close_filestream(file);
	return samp;
}

void free_sample(sample* samp) {
	xfree(samp->bboxes);
	xfree(samp);
}

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
		size_t length = tokens_length(tokens);
		if (strcmp(tokens[0], "data_directory") == 0) {
			dp->data_dir = tokens[1];
		}
		else if (strcmp(tokens[0], "images_folder") == 0) {
			dp->imgs_dir = tokens[1];
		}
		else if (strcmp(tokens[0], "annotations_folder") == 0) {
			dp->ants_dir = tokens[1];
		}
		else if (strcmp(tokens[0], "classes_file") == 0) {
			dp->classes_file = tokens[1];
		}
		else if (strcmp(tokens[0], "backup_folder") == 0) {
			dp->backup_dir = tokens[1];
		}
		else if (strcmp(tokens[0], "weights_file") == 0) {
			dp->weights_file = tokens[1];
		}
		else if (strcmp(tokens[0], "cfg_file") == 0) {
			dp->cfg_file = tokens[1];
		}
	}
	return dp;
}

void free_data_paths(data_paths* dp) {
	xfree(dp->ants_dir);
	xfree(dp->backup_dir);
	xfree(dp->cfg_file);
	xfree(dp->classes_file);
	xfree(dp->data_dir);
	xfree(dp->imgs_dir);
	xfree(dp->weights_file);
	xfree(dp);
}

void get_sample_files(char* imgdir, char* antdir) {

}