#ifndef XOPENCV_H
#define XOPENCV_H


#ifdef __cplusplus
extern "C" {
#endif

#ifdef OPENCV
	void show_image_opencv(float* data, int w, int h, int c, int waitkey);
	float* load_image_opencv(char* filename, size_t* w, size_t* h, size_t* c);
#endif

#ifdef __cplusplus
}
#endif
#endif