#ifndef DATA_H
#define DATA_H


#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif


	typedef struct bbox bbox;
	typedef struct sample sample;
	typedef struct data_paths;

	sample* load_samples(char* folder);
	sample* load_sample(char* txtfile, char* imgfile);
	void free_sample(sample* samp);


	typedef struct bbox {
		int lbl;
		// values are relative
		float cx;
		float cy;
		float w;
		float h;
	} bbox;

	typedef struct sample {
		size_t nboxes;
		bbox* bboxes;
		char* imgpath;
	} sample;

	typedef struct data_paths {
		char* data_dir;
		char* images_dir;
		char* annots_dir;
		char* classes_file;
		char* backup_dir;
		char* weights_file;
		char* cfg_file;
	} data_paths;

#ifdef __cplusplus
}
#endif
#endif
