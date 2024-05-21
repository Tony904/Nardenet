#include "data.h"
#include <assert.h>
#include <string.h>
#include "utils.h"
#include "xallocs.h"
#include "image.h"


#define MAX_DIR_PATH 255U
#define MIN_FILENAME_LENGTH 5U


sample* load_samples(char* directory) {
	size_t dirlen = strlen(directory);
	if (dirlen >= MAX_DIR_PATH) {
		printf("Directory path too long. Must be less than %zu characters.\npath: %s\ncharacters:%zu\n", MAX_DIR_PATH + 1U, directory, dirlen);
		wait_for_key_then_exit();
	}
	char* dir[MAX_DIR_PATH] = { 0 };
	strcpy(dir, directory);
	if (dir[dirlen - 1] != '\\') {
		dir[dirlen] = '\\';
		dirlen++;
	}
	list* imgpaths = get_files_list(dir, ".bmp,.jpg,.jpeg,.png");
	size_t nimgs = imgpaths->length;
	sample* samples = (sample*)xcalloc(nimgs, sizeof(sample));
	node* noed;
	for (size_t i = 0; i < nimgs; i++) {
		if (i == 0) noed = imgpaths->first;
		else noed = noed->next;
		char* imgfile = (char*)noed->val;
		char* antfile[MAX_DIR_PATH + MIN_FILENAME_LENGTH] = { 0 };
		char* dot = strrchr(imgfile, '.');
		memcpy(antfile, imgfile, strlen(imgfile) - strlen(dot));
		memcpy(&antfile[strlen(antfile)], ".txt", 4U);
		load_sample(antfile, imgfile, &samples[i]);
	}
	free_list(imgpaths);
}

void load_sample(char* antfile, char* imgfile, sample* samp) {
	if (!file_exists(imgfile)) wait_for_key_then_exit();
	FILE* file = get_filestream(antfile, "r");
	size_t n_lines = get_line_count(file);
	samp->nboxes = n_lines;
	samp->imgpath = (char*)xcalloc(strlen(imgfile) + 1, sizeof(char));
	strcpy(samp->imgpath, imgfile);
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