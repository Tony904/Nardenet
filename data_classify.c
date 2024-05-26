#include "data_classify.h"
#include <string.h>
#include "xallocs.h"
#include "utils.h"


#define MAX_DIR_PATH 255
#define MIN_FILENAME_LENGTH 5


class_set* load_class_sets(char* class_dir, char** class_names, size_t n_classes) {
	class_set* sets = (class_set*)xcalloc(n_classes, sizeof(class_set));
	for (size_t i = 0; i < n_classes; i++) {
		load_class_set(&sets[i], class_dir, class_names[i]);
	}
}

void load_class_set(class_set* set, char* class_dir, char* class_name) {
	size_t dirlen = strlen(class_dir);
	if (dirlen >= MAX_DIR_PATH) {
		printf("Directory path too long. Must be less than %d characters.\npath: %s\ncharacters:%zu\n", MAX_DIR_PATH + 1, directory, dirlen);
		wait_for_key_then_exit();
	}
	char dir[MAX_DIR_PATH] = { 0 };
	strcpy(dir, class_dir);
	if (dir[dirlen - 1] != '\\' && dirlen < MAX_DIR_PATH) {
		dir[dirlen] = '\\';
		dirlen++;
	}





	list* imgpaths = get_files_list(dir, ".bmp,.jpg,.jpeg,.png");
	size_t n_samps = imgpaths->length;
	det_sample* samples = (det_sample*)xcalloc(n_samps, sizeof(det_sample));
	node* noed = { 0 };
	for (size_t i = 0; i < n_samps; i++) {
		if (i == 0) noed = imgpaths->first;
		else noed = noed->next;
		char* imgfile = (char*)noed->val;
		char antfile[MAX_DIR_PATH + MIN_FILENAME_LENGTH] = { 0 };
		char* dot = strrchr(imgfile, '.');
		memcpy(antfile, imgfile, strlen(imgfile) - strlen(dot));
		memcpy(&antfile[strlen(antfile)], ".txt", 4U);
		load_det_sample(antfile, imgfile, &samples[i]);
	}
	*count_dst = n_samps;
	free_list(imgpaths);
	return samples;
}