#ifndef DATA_CLASSIFIER_H
#define DATA_CLASSIFIER_H


#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif

	typedef struct class_set class_set;
	typedef struct classifier_dataset classifier_dataset;

	class_set* load_class_sets(char* classes_dir, char** class_names, size_t n_classes);
	void free_class_sets(class_set* set, size_t n);

	typedef struct class_set {  // classifier class dataset
		char** files;
		size_t class_id;
		size_t n;  // # of files
		size_t* rands;  // array of non-repeating random numbers of size n
		size_t ri;  // rands index
	} class_set;

	typedef struct classifier_dataset {
		class_set* sets;
		size_t n;  // # of sets/classes
		size_t* rands;  // array of non-repeating random numbers of size n
		size_t ri;  // rands index
	} classifier_dataset;

#ifdef __cplusplus
}
#endif
#endif