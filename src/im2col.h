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
	void wgrads2im_cpu(float* wgrads,
		int channels, int height, int width,
		int ksize, int pad, int stride,
		float* im);
	void sum_columns(int rows, int cols, float* data, float* sums);
	void test_wgrads2im(void);
	void pprint_mat(float* data, int width, int height, int channels);
	void test_sum_columns(void);

#ifdef __cplusplus
}
#endif
#endif