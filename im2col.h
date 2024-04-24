#ifndef IM2COL_H
#define IM2COL_H


#ifdef __cplusplus
extern "C" {
#endif

	void img2col(float* img, int width, int height, int channels,
		int ksize, int pad, int stride,
		float* dst);
	void test_img2col(void);


#ifdef __cplusplus
}
#endif
#endif