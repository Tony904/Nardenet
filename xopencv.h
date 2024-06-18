#ifndef XOPENCV_H
#define XOPENCV_H


#include "image.h"


#ifdef __cplusplus
extern "C" {
#endif

#ifdef OPENCV
	image* load_file_to_image(char* filename);
	void show_image(image* img);
#endif

#ifdef __cplusplus
}
#endif
#endif