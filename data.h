#ifndef DATA_H
#define DATA_H


#ifdef __cplusplus
extern "C" {
#endif


	sample* load_samples(char* folder);
	sample* load_sample(char* txtfile, char* imgfile);
	void free_sample(sample* samp);


	typedef struct bbox bbox;
	typedef struct sample sample;


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


#ifdef __cplusplus
}
#endif
#endif
