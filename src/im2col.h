#ifndef IM2COL_H
#define IM2COL_H


#ifdef __cplusplus
extern "C" {
#endif

	void im2col(float* data_im, int channels,
		int height, int width, int ksize,
		int pad, int stride,
		float* data_col);
	void col2im(float* data_col,
		int channels, int height, int width,
		int kernel_size, int pad, int stride,
		float* data_im);
	void test_col2im(void);
	void test_im2col(void);

#ifdef __cplusplus
}
#endif
#endif