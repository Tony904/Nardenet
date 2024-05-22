#ifndef DATA_H
#define DATA_H


#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif


	typedef struct bbox bbox;
	typedef struct sample sample;
	typedef struct data_paths data_paths;

	sample* load_samples(char* directory, size_t* count_dst);
	void load_sample(char* antfile, char* imgfile, sample*);
	data_paths* get_data_paths(char* datafile);
	void free_sample(sample* samp);
	void free_data_paths(data_paths* dp);
	void print_samples(sample* samples, size_t count, int print_annotations);


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
		char* imgs_dir;
		char* ants_dir;
		char* classes_file;
		char* backup_dir;
		char* weights_file;
		char* cfg_file;
	} data_paths;

#ifdef __cplusplus
}
#endif
#endif
