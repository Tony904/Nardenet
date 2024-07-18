#ifndef DATA_CLASSIFIER_H
#define DATA_CLASSIFIER_H


#include <stdio.h>
#include "network.h"


#ifdef __cplusplus
extern "C" {
#endif
	
	class_set* load_class_sets(char* classes_dir, char** class_names, size_t n_classes);
	void free_class_sets(class_set* set, size_t n);

#ifdef __cplusplus
}
#endif
#endif