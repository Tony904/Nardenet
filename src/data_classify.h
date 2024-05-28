#ifndef DATA_CLASSIFIER_H
#define DATA_CLASSIFIER_H


#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif

	typedef struct class_set class_set;

	class_set* load_class_sets(char* classes_dir, char** class_names, size_t n_classes);

	typedef struct class_set {  // classifier class dataset
		int class_id;
		char** files;
		size_t n;  // # of files
		size_t* rands;  // array of non-repeating random numbers of size n
		size_t ri;  // rands index
	} class_set;

#ifdef __cplusplus
}
#endif
#endif