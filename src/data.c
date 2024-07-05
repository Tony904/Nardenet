#include "data.h"
#include <string.h>
#include "utils.h"
#include "xallocs.h"


class_set* classifier_dataset_get_next_rand_class_set(classifier_dataset* dataset);
image* class_set_get_next_rand_image(class_set* set);


image* get_next_image_classifier_dataset(classifier_dataset* dataset, float* truth) {
	class_set* set = classifier_dataset_get_next_rand_class_set(dataset);
	size_t n = dataset->n;
	for (size_t i = 0; i < n; i++) { truth[i] = 0.0F; }
	truth[set->class_id] = 1.0F;
	return class_set_get_next_rand_image(set);
}

class_set* classifier_dataset_get_next_rand_class_set(classifier_dataset* dataset) {
	size_t ri = dataset->ri;
	size_t* rands = dataset->rands;
	size_t n_sets = dataset->n;
	class_set* sets = dataset->sets;
	class_set* set = &sets[rands[ri]];
	ri++;
	if (!(ri < n_sets)) {
		ri = 0;
		get_random_numbers_no_repeats(rands, n_sets, 0, n_sets - 1);
	}
	dataset->ri = ri;
	return set;
}

image* class_set_get_next_rand_image(class_set* set) {
	size_t ri = set->ri;
	size_t* rands = set->rands;
	size_t n_sets = set->n;
	image* img = load_image(set->files[rands[ri]]);
	printf("%s\n", set->files[rands[ri]]);
	ri++;
	if (!(ri < n_sets)) {
		ri = 0;
		get_random_numbers_no_repeats(rands, n_sets, 0, n_sets - 1);
	}
	set->ri = ri;
	return img;
}

void load_classifier_dataset(classifier_dataset* dataset, char* classes_dir, char** class_names, size_t n_classes) {
	dataset->sets = load_class_sets(classes_dir, class_names, n_classes);
	dataset->rands = (size_t*)xcalloc(n_classes, sizeof(size_t));
	dataset->n = n_classes;
	dataset->ri = 0;
	get_random_numbers_no_repeats(dataset->rands, n_classes, 0, n_classes - 1);
}

void free_classifier_dataset_members(classifier_dataset* dataset) {
	free_class_sets(dataset->sets, dataset->n);
	xfree(dataset->rands);
	dataset->n = 0;
	dataset->ri = 0;
	dataset->rands = 0;
	dataset->sets = 0;
}
