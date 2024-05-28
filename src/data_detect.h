#ifndef DATA_DETECT_H
#define DATA_DETECT_H


#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif

	typedef struct bbox bbox;
	typedef struct det_sample det_sample;

	det_sample* load_det_samples(char* directory, size_t* count_dst);
	void free_det_sample(det_sample* samp);
	void print_det_samples(det_sample* samples, size_t count, int print_annotations);

	typedef struct bbox {
		int lbl;
		// values are relative
		float cx;
		float cy;
		float w;
		float h;
	} bbox;

	typedef struct det_sample {  // object detection sample
		size_t nboxes;
		bbox* bboxes;
		char* imgpath;
	} det_sample;

#ifdef __cplusplus
}
#endif
#endif