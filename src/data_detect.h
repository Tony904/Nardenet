#ifndef DATA_DETECT_H
#define DATA_DETECT_H


#include <stdio.h>
#include "network.h"


#ifdef __cplusplus
extern "C" {
#endif

	det_sample* load_det_samples(char* directory, size_t* count_dst);
	void free_det_sample(det_sample* samp);
	void print_det_samples(det_sample* samples, size_t count, int print_annotations);

#ifdef __cplusplus
}
#endif
#endif