#ifndef IM2COL_H
#define IM2COL_H


#ifdef __cplusplus
extern "C" {
#endif

	void im2col(float* data_im, float* data_col,
		int im_w, int im_h, int im_channels,
		int out_w, int out_h,
		int ksize, int stride, int pad);
	void col2im(float* data_col, float* data_im,
		int im_w, int im_h, int im_channels,
		int out_w, int out_h,
		int ksize, int stride, int pad);
	void test_col2im(void);
	void test_im2col(void);

#ifdef __cplusplus
}
#endif
#endif