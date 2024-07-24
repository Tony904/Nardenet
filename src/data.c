#include "data.h"
#include <string.h>
#include "utils.h"
#include "xallocs.h"
#include "image.h"
#include "data_classify.h"
#include "data_detect.h"
#include "iou.h"
#include "math.h"


void classifier_dataset_get_next_image(classifier_dataset* dataset, image* dst, float* truth);


void detector_dataset_get_next_image(detector_dataset* dataset, image* dst, float* truth) {
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

	// get image from the selected sample
	load_image_to_buffer(sample->imgpath, dst);

}

float* generate_detect_layer_truth(network* net, layer* l, det_sample* samples, size_t batch_size) {
	det_cell* cells = l->cells;
	LOSS_TYPE loss = l->loss_type;
	bbox* anchors = l->anchors;
	size_t n_anchors = l->n_anchors;
	size_t net_w = net->w;
	size_t net_h = net->h;
	size_t l_w = l->w;
	size_t l_h = l->h;
	size_t l_wh = l_w * l_h;
	size_t n_classes = l->n_classes;
#pragma omp parallel for firstprivate(n_anchors)
	for (size_t i = 0; i < l_wh; i++) {
		for (size_t j = 0; j < n_anchors; j++) {
			cells[j].obj = 0;
			cells[j].cls = 0;
		}
	}
	// output format PER ANCHOR: {any_object, cx, cy, w, h, class1, class2, class3, etc...}
	// output format general: {batch1_cell1_any_obj, batch1_cell1_cx,... batch1_cell2... batch2_cell1... etc...}
	// note: bbox coordinates are offsets from top-left (0, 0) of grid cell
	if (net_w != net_h || l_w != l_h) {
		printf("Network input and output width & height must be square.\n");
		wait_for_key_then_exit();
	}
	float cell_size = (float)l_w / (float)net_w;
	for (size_t i = 0; i < l_wh; i++) {
		float top = cells[i].top;  // values are relative to % of image dimensions
		float left = cells[i].left;
		float bottom = cells[i].bottom;
		float right = cells[i].right;
		int* cell_obj = cells[i].obj;
		int* cell_cls = cells[i].cls;
		bbox* cell_tboxes = cells[i].tboxes;
#pragma omp parallel for firstprivate(top, left, bottom, right, i, n_anchors, cell_obj, cell_cls, cell_tboxes)
		for (size_t b = 0; b < batch_size; b++) {
			det_sample* sample = &samples[b];
			int bn = (int)(b * n_anchors);
			for (size_t x = 0; x < sample->n; x++) {
				bbox box = sample->bboxes[x];
				if (box.cx < left) continue;
				if (box.cx >= right) continue;
				if (box.cy < top) continue;
				if (box.cy >= bottom) continue;
				int best_i = -1;
				float best_iou = 0.0F;
				for (size_t j = 0; j < n_anchors; j++) {
					float iou = get_ciou(box, anchors[j]);
					if (iou > best_iou) {
						if (cells[i].obj[j]) continue;
						best_iou = iou;
						best_i = j;
					}
				}
				if (best_i < 0) continue;
				size_t index = (size_t)(bn * best_i);
				cell_obj[index] = 1;
				cell_cls[index] = box.lbl;
				cell_tboxes[index].cx = box.cx;
				cell_tboxes[index].cy = box.cy;
				cell_tboxes[index].w = box.w;
				cell_tboxes[index].h = box.h;
				cell_tboxes[index].area = box.area;
				cell_tboxes[index].left = box.left;
				cell_tboxes[index].right = box.right;
				cell_tboxes[index].top = box.top;
				cell_tboxes[index].bottom = box.bottom;
			}
		}
	}
}

void load_detector_dataset(detector_dataset* dataset, char* dir) {
	size_t count;
	dataset->samples = load_det_samples(dir, &count);
	dataset->n = count;
	dataset->rands = (size_t*)xcalloc(count, sizeof(size_t));
	dataset->ri = 0;
	get_random_numbers_no_repeats(dataset->rands, count, 0, count - 1);
}

void classifier_get_next_batch(classifier_dataset* dataset, size_t batch_size, float* data, size_t w, size_t h, size_t c, float* truth, size_t n_classes) {
	size_t n = w * h * c;
	for (size_t s = 0; s < batch_size; s++) {
		image img = { 0 };
		img.w = w;
		img.h = h;
		img.c = c;
		img.data = &data[s * n];
		classifier_dataset_get_next_image(dataset, &img, &truth[s * n_classes]);
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
	if (ri >= n) {
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
	load_image_to_buffer(set->files[rands[ri]], dst);
	//printf("%s\n", set->files[rands[ri]]);
	ri++;
	if (!(ri < n)) {
		ri = 0;
		get_random_numbers_no_repeats(rands, n, 0, n - 1);
	}
	set->ri = ri;
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
