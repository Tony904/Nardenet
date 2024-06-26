#ifndef IM2COL_H
#define IM2COL_H


#ifdef __cplusplus
extern "C" {
#endif

	float* im2col_cpu(const float* data_im, 
		const int channels, const int height, const int width, 
		const int ksize, const int pad, const int stride,
		float* data_col);
	void test_im2col(void);
	void col2im_cpu(const float* data_col,
		int channels, int height, int width,
		int kernel_size, int pad, int stride,
		float* data_im);
	void sum_columns(int rows, int cols, float* data, float* sums);
	void test_col2im(void);
	void pprint_mat(float* data, int width, int height, int channels);

#ifdef __cplusplus
}
#endif
#endif