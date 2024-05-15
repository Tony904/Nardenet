#include "data.h"
#include <assert.h>
#include <string.h>
#include "utils.h"
#include "xallocs.h"
#include "image.h"


sample* load_samples(char* folder) {

}

sample* load_sample(char* txtfile, char* imgfile) {
	if (!file_exists(imgfile)) wait_for_key_then_exit();
	FILE* file = get_filestream(txtfile, "r");
	size_t n_lines = get_line_count(txtfile);
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
			printf("Line %zu, file %s\n", n, txtfile);
			wait_for_key_then_exit();
		}
		size_t length = tokens_length(tokens);
		if (length != 5) {
			printf("Bounding boxes must have 5 parameters. Has %zu.\n", length);
			printf("Line %zu, file %s\n", n, txtfile);
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
