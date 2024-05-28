#ifndef IM2COL_H
#define IM2COL_H


#ifdef __cplusplus
extern "C" {
#endif

	float* im2col_cpu(const float* data_im, const int channels,
		const int height, const int width, const int ksize,
		const int pad, const int stride,
		float* data_col);
	void test_im2col(void);

#ifdef __cplusplus
}
#endif
#endif