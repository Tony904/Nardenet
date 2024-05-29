#ifndef DATA_H
#define DATA_H


#include <stdio.h>
#include "data_detect.h"
#include "data_classify.h"
#include "image.h"


#ifdef __cplusplus
extern "C" {
#endif

	typedef struct classifier_dataset classifier_dataset;
	typedef struct detector_dataset detector_dataset;

	void load_classifier_dataset(classifier_dataset* dataset, char* classes_dir, char** class_names, size_t n_classes);
	void load_detector_dataset(detector_dataset* dataset);
	image* get_next_image_classifier_dataset(classifier_dataset* dataset, float* truth);

	typedef struct classifier_dataset {
		class_set* sets;
		size_t n;  // # of sets/classes
		size_t* rands;  // array of non-repeating random numbers of size n
		size_t ri;  // rands index
	} classifier_dataset;

	typedef struct detector_dataset {
		det_sample* samples;
		size_t n;  // # of samples
		size_t* rands;  // array of non-repeating random numbers of size n
		size_t ri;  // rands index
	} detector_dataset;
	
#ifdef __cplusplus
}
#endif
#endif
