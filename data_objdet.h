#ifndef DATA_OBJDET_H
#define DATA_OBJDET_H


#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif


	od_sample* load_od_samples(char* directory, size_t* count_dst);
	void free_od_sample(od_sample* samp);
	void print_od_samples(od_sample* samples, size_t count, int print_annotations);

	typedef struct bbox bbox;
	typedef struct od_sample od_sample;

	typedef struct bbox {
		int lbl;
		// values are relative
		float cx;
		float cy;
		float w;
		float h;
	} bbox;

	typedef struct od_sample {  // object detection sample
		size_t nboxes;
		bbox* bboxes;
		char* imgpath;
	} od_sample;


#ifdef __cplusplus
}
#endif
#endif