#ifndef DATA_H
#define DATA_H


#include <stdio.h>
#include "data_objdet.h"
#include "data_classifier.h"


#ifdef __cplusplus
extern "C" {
#endif


	typedef enum DATASET_TYPE DATASET_TYPE;

	typedef struct bbox bbox;
	typedef struct od_sample od_sample;
	typedef struct class_set class_set;
	typedef struct dataset dataset;
	typedef struct data_paths data_paths;

	typedef enum DATASET_TYPE {
		DATASET_CLASSIFY,
		DATASET_OD
	} DATASET_TYPE;

	typedef struct dataset {
		DATASET_TYPE type;
		size_t n;  // # of samples or # of sets
		union data {
			class_set* sets;
			od_sample* samples;
		} data;
	} dataset;

	
#ifdef __cplusplus
}
#endif
#endif
