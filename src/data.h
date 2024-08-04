#ifndef DATA_H
#define DATA_H


#include <stdio.h>
#include "network.h"


#ifdef __cplusplus
extern "C" {
#endif

	void detector_get_next_batch(network* net);
	void load_detector_dataset(detector_dataset* dataset, char* dir);
	void classifier_get_next_batch(classifier_dataset* dataset, size_t batch_size, float* data, size_t w, size_t h, size_t c, float* truth, size_t n_classes);
	void load_classifier_dataset(classifier_dataset* dataset, char* classes_dir, char** class_names, size_t n_classes);
	void free_classifier_dataset_members(classifier_dataset* dataset);

#ifdef __cplusplus
}
#endif
#endif
