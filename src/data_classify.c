#include "data_classify.h"
#include <string.h>
#include "xallocs.h"
#include "utils.h"
#include "list.h"


#define MAX_DIR_PATH _MAX_PATH - 5


void load_class_set(class_set* set, char* class_dir);


class_set* load_class_sets(char* classes_dir, char** class_names, size_t n_classes) {
	class_set* sets = (class_set*)xcalloc(n_classes, sizeof(class_set));
	char buff[MAX_DIR_PATH] = { 0 };
	for (int i = 0; i < n_classes; i++) {
		snprintf(buff, sizeof(buff), "%s%s\\", classes_dir, class_names[i]);
		sets[i].class_id = i;
		load_class_set(&sets[i], buff);
		print_str_array(sets[i].files, sets[i].n);
		printf("\n");
	}
	return sets;
}

void load_class_set(class_set* set, char* class_dir) {
	size_t dirlen = strlen(class_dir);
	if (dirlen >= MAX_DIR_PATH) {
		printf("Directory path too long. Must be less than %d characters.\npath: %s\ncharacters:%zu\n", MAX_DIR_PATH + 1, class_dir, dirlen);
		wait_for_key_then_exit();
	}
	char dir[MAX_DIR_PATH] = { 0 };
	strcpy(dir, class_dir);
	list* lst = get_files_list(dir, IMG_EXTS);
	char** files = (char**)xcalloc(lst->length, sizeof(char*));
	node* noed = lst->first;
	for (size_t i = 0; i < lst->length; i++) {
		files[i] = (char*)noed->val;
		noed = noed->next;
	}
	set->n = lst->length;
	set->files = files;
	set->rands = (size_t*)xcalloc(set->n, sizeof(size_t));
	get_random_numbers_no_repeats(set->rands, set->n, 0, set->n - 1);
	free_list(lst, 0);
}

void free_class_sets(class_set* sets, size_t n) {
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < sets[i].n; j++) {
			xfree(sets[i].files[j]);
		}
		xfree(sets[i].files);
		xfree(sets[i].rands);
		sets[i].class_id = 0;
		sets[i].files = 0;
		sets[i].n = 0;
		sets[i].rands = 0;
		sets[i].ri = 0;
	}
	xfree(sets);
}