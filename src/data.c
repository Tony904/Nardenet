#include "data.h"
#include <string.h>
#include <assert.h>
#include "utils.h"
#include "xallocs.h"
#include "image.h"
#include "data_classify.h"
#include "data_detect.h"
#include "iou.h"
#include "math.h"


det_sample* detector_dataset_get_next_sample(detector_dataset* dataset, image* dst);
void classifier_dataset_get_next_image(classifier_dataset* dataset, image* dst, float* truth);


void detector_get_next_batch(network* net) {
	size_t batch_size = net->batch_size;
	detector_dataset* dataset = &net->data.detr;
	size_t w = net->w;
	size_t h = net->h;
	size_t c = net->c;
	size_t n = w * h * c;
	det_sample** batch_samples = net->data.detr.current_batch;
	float brightness_lower = net->exposure[0];
	float brightness_upper = net->exposure[1];
	float contrast_lower = 1.0F;
	float contrast_upper = 1.0F;
	float saturation_lower = net->saturation[0];
	float saturation_upper = net->saturation[1];
	float hue_lower = net->hue[0];
	float hue_upper = net->hue[1];
	for (size_t b = 0; b < batch_size; b++) {
		image img = { 0 };
		img.w = w;
		img.h = h;
		img.c = c;
		img.data = &net->input->output[b * n];
		batch_samples[b] = detector_dataset_get_next_sample(dataset, &img);
		prescale_image(&img);
		randomize_colorspace(&img, brightness_lower, brightness_upper, contrast_lower, contrast_upper, saturation_lower, saturation_upper, hue_lower, hue_upper);
	}
}

det_sample* detector_dataset_get_next_sample(detector_dataset* dataset, image* dst) {
	// get next sample (based on array of random numbers)
	size_t ri = dataset->ri;
	size_t* rands = dataset->rands;
	size_t n = dataset->n;  // # of samples
	det_sample* sample = &dataset->samples[ri];
	ri++;

	// generate new random numbers if at end of rands array
	if (ri >= n) {
		ri = 0;
		get_random_numbers_no_repeats(rands, n, 0, n - 1);
	}
	dataset->ri = ri;
	printf("Loading sample: %s\n", sample->imgpath);
	// get image from the selected sample
	load_image_to_buffer(sample->imgpath, dst, 1);
	return sample;
}

void load_detector_dataset(detector_dataset* dataset, char* dir) {
	size_t count;
	dataset->samples = load_det_samples(dir, &count);
	dataset->n = count;
	dataset->rands = (size_t*)xcalloc(count, sizeof(size_t));
	dataset->ri = 0;
	get_random_numbers_no_repeats(dataset->rands, count, 0, count - 1);
}

void classifier_get_next_batch(network* net) {
	classifier_dataset* dataset = &net->data.clsr;
	float* truth = net->layers[net->n_layers - 1].truth;
	size_t batch_size = net->batch_size;
	size_t n_classes = net->n_classes;
	float* data = net->input->output;
	size_t w = net->w;
	size_t h = net->h;
	size_t c = net->c;
	size_t n = w * h * c;
	float brightness_lower = net->exposure[0];
	float brightness_upper = net->exposure[1];
	float contrast_lower = 1.0F;
	float contrast_upper = 1.0F;
	float saturation_lower = net->saturation[0];
	float saturation_upper = net->saturation[1];
	float hue_lower = net->hue[0];
	float hue_upper = net->hue[1];
	for (size_t s = 0; s < batch_size; s++) {
		image img = { 0 };
		img.w = w;
		img.h = h;
		img.c = c;
		img.data = &data[s * n];
		classifier_dataset_get_next_image(dataset, &img, &truth[s * n_classes]);
		randomize_colorspace(&img, brightness_lower, brightness_upper, contrast_lower, contrast_upper, saturation_lower, saturation_upper, hue_lower, hue_upper);
	}
}

void classifier_dataset_get_next_image(classifier_dataset* dataset, image* dst, float* truth) {
	// get next class_set (based on array of random numbers)
	size_t ri = dataset->ri;
	size_t* rands = dataset->rands;
	size_t n = dataset->n;  // # of class_sets in dataset.sets
	class_set* sets = dataset->sets;
	class_set* set = &sets[rands[ri]];
	ri++;
	// generate new random numbers if at end of rands array
	if (!(ri < n)) {
		ri = 0;
		get_random_numbers_no_repeats(rands, n, 0, n - 1);
	}
	dataset->ri = ri;

	// set truth vector
	for (size_t i = 0; i < dataset->n; i++) { truth[i] = 0.0F; }
	truth[set->class_id] = 1.0F;

	// get next image from the selected class_set (also based on array of random numbers)
	ri = set->ri;
	rands = set->rands;
	n = set->n;  // # of files in class_set.files
	load_image_to_buffer(set->files[rands[ri]], dst, 1);
	//printf("%s\n", set->files[rands[ri]]);
	ri++;
	if (!(ri < n)) {
		ri = 0;
		get_random_numbers_no_repeats(rands, n, 0, n - 1);
	}
	set->ri = ri;
}

void load_classifier_dataset(classifier_dataset* dataset, char* classes_dir, char** class_names, size_t n_classes, char* interpath) {
	dataset->sets = load_class_sets(classes_dir, class_names, n_classes, interpath);
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
