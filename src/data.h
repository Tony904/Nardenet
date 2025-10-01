#ifndef DATA_H
#define DATA_H


#include <stdio.h>
#include "network.h"


#ifdef __cplusplus
extern "C" {
#endif

	void detector_get_next_batch(network* net);
	void load_detector_dataset(detector_dataset* dataset, char* dir);
	void classifier_get_next_batch(network* net);
	void load_classifier_dataset(classifier_dataset* dataset, char* classes_dir, char** class_names, size_t n_classes, char* interpath);
	void free_classifier_dataset_fields(classifier_dataset* dataset);

#ifdef __cplusplus
}
#endif
#endif
