#ifndef DATA_CLASSIFIER_H
#define DATA_CLASSIFIER_H


#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif


	typedef struct class_set class_set;

	typedef struct class_set {  // classifier class dataset
		int class_id;
		char** files;
		size_t n;  // # of files
	} class_set;


#ifdef __cplusplus
}
#endif
#endif