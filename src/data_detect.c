#include "data_detect.h"
#include <string.h>
#include "list.h"
#include "xallocs.h"
#include "utils.h"


#define MAX_DIR_PATH _MAX_PATH - 5


void load_det_sample(char* antfile, char* imgfile, det_sample* samp);
void print_det_sample(det_sample s, int print_annotations);
void print_bboxes(bbox* boxes, size_t count);
void print_bbox(bbox b);


det_sample* load_det_samples(char* directory, size_t* count_dst) {
	size_t dirlen = strlen(directory);
	if (dirlen >= MAX_DIR_PATH) {
		printf("Directory path too long. Must be less than %d characters.\npath: %s\ncharacters:%zu\n", MAX_DIR_PATH + 1, directory, dirlen);
		wait_for_key_then_exit();
	}
	char dir[MAX_DIR_PATH] = { 0 };
	strcpy(dir, directory);
	if (dir[dirlen - 1] != '\\' && dirlen < MAX_DIR_PATH) {
		dir[dirlen] = '\\';
		dirlen++;
	}
	list* imgpaths = get_files_list(dir, IMG_EXTS);
	size_t n_samps = imgpaths->length;
	det_sample* samples = (det_sample*)xcalloc(n_samps, sizeof(det_sample));
	node* noed = { 0 };
	for (size_t i = 0; i < n_samps; i++) {
		if (i == 0) noed = imgpaths->first;
		else noed = noed->next;
		char* imgfile = (char*)noed->val;
		char antfile[_MAX_PATH] = { 0 };
		char* dot = strrchr(imgfile, '.');
		memcpy(antfile, imgfile, strlen(imgfile) - strlen(dot));
		memcpy(&antfile[strlen(antfile)], ".txt", 4);
		load_det_sample(antfile, imgfile, &samples[i]);
		bbox asdf = samples[i].bboxes[0];
		print_bbox(asdf);
	}
	*count_dst = n_samps;
	free_list(imgpaths, 0);
	return samples;
}

void load_det_sample(char* antfile, char* imgfile, det_sample* samp) {
	if (!file_exists(imgfile)) wait_for_key_then_exit();
	FILE* file = get_filestream(antfile, "r");
	size_t n_lines = get_line_count(file);
	samp->n = n_lines;  // # of bboxes
	samp->imgpath = (char*)xcalloc(strlen(imgfile) + 1, sizeof(char));
	strcpy(samp->imgpath, imgfile);
	samp->bboxes = (bbox*)xcalloc(n_lines, sizeof(bbox));
	char* line;
	size_t n = 0;
	while (line = read_line(file), line != 0) {
		//clean_string(line);
		char** tokens = split_string(line, " ");
		if (!tokens) {
			printf("Error parsing bounding box string.\n");
			printf("Line %zu, file %s\n", n, antfile);
			wait_for_key_then_exit();
		}
		size_t length = tokens_length(tokens);
		if (length != 5) {
			printf("Bounding boxes must have 5 parameters, has %zu.\n", length);
			printf("Line %zu, file %s\n", n, antfile);
			wait_for_key_then_exit();
		}
		bbox* box = &samp->bboxes[n];
		box->lbl = str2int(tokens[0]);
		box->cx = str2float(tokens[1]);
		box->cy = str2float(tokens[2]);
		box->w = str2float(tokens[3]);
		box->h = str2float(tokens[4]);
		box->prob = 1.0F;
		box->area = box->w * box->h;
		box->left = box->cx - (box->w * 0.5F);
		box->top = box->cy - (box->h * 0.5F);
		box->right = box->cx + (box->w * 0.5F);
		box->bottom = box->cy + (box->w * 0.5F);
		xfree(&tokens);
		n++;
	}
	close_filestream(file);
}

void free_det_sample(det_sample* samp) {
	xfree(&samp->bboxes);
	xfree(&samp);
}

void print_det_samples(det_sample* samples, size_t count, int print_annotations) {
	printf("\n[SAMPLES]\ncount: %zu\n\n", count);
	for (size_t i = 0; i < count; i++) {
		print_det_sample(samples[i], print_annotations);
		printf("\n");
	}
	printf("[END]\n\n");
}

void print_det_sample(det_sample s, int print_annotations) {
	printf("File: %s\n", s.imgpath);
	printf("# of bboxes: %zu\n", s.n);
	if (print_annotations) print_bboxes(s.bboxes, s.n);
}

void print_bboxes(bbox* boxes, size_t count) {
	for (size_t i = 0; i < count; i++) {
		print_bbox(boxes[i]);
	}
}

void print_bbox(bbox b) {
	printf("lbl: %d  cx: %f  cy: %f  w: %f  h: %f\n", b.lbl, b.cx, b.cy, b.w, b.h);
}
