#ifndef NARDENET_STBIMAGE_H
#define NARDENET_STBIMAGE_H

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


	float* load_image_stbi(char* filename, size_t* w, size_t* h, size_t* c);
	void write_image_stbi(char* filename, float* data, int w, int h, int c);


#ifdef __cplusplus
}
#endif
#endif