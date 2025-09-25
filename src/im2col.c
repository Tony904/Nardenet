#include "im2col.h"
#include <stdlib.h>
#include "xallocs.h"
#include "gemm.h"
#include "utils.h"


// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
inline static int is_a_ge_zero_and_a_lt_b(int aa, int bb) {
	return (unsigned)(aa) < (unsigned)(bb);
}

void im2col(float* data_im, float* data_col,
	int im_w, int im_h, int im_channels,
	int out_w, int out_h,
	int ksize, int stride, int pad)
{
	int out_wh = out_w * out_h;
	int im_wh = im_w * im_h;
	int ch;
#pragma omp parallel for firstprivate(im_w, im_h, out_w, out_h, ksize, pad, stride)
	for (ch = 0; ch < im_channels; ch++) {
		float* im = &data_im[ch * im_wh];
		float* cm = &data_col[ch * out_wh * ksize * ksize];
		for (int krow = 0; krow < ksize; krow++) {
			for (int kcol = 0; kcol < ksize; kcol++) {
				int im_row = krow - pad;
				for (int out_rows = out_h; out_rows; out_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(im_row, im_h)) {
						for (int out_cols = out_w; out_cols; out_cols--) {
							*(cm++) = 0;
						}
					}
					else {
						int im_col = kcol - pad;
						for (int out_cols = out_w; out_cols; out_cols--) {
							if (is_a_ge_zero_and_a_lt_b(im_col, im_w)) {
								*(cm++) = im[im_row * im_w + im_col];
							}
							else {
								*(cm++) = 0;
							}
							im_col += stride;
						}
					}
					im_row += stride;
				}
			}
		}
	}
}

void test_im2col(void) {
	int width = 3;
	int height = 3;
	int channels = 2;
	int img_size = width * height * channels;
	float* img = (float*)xcalloc(img_size, sizeof(float));
	int pad = 0;
	int stride = 1;
	int ksize = 2;
	int out_w = (width + pad * 2 - ksize) / stride + 1;
	int out_h = (height + pad * 2 - ksize) / stride + 1;
	int dst_w = out_w * out_h;
	int dst_h = ksize * ksize * channels;
	int dst_size = dst_w * dst_h;
	float* dst = (float*)xcalloc(dst_size * 2, sizeof(float));

	for (int i = 0; i < img_size; i++) {
		img[i] = (float)(i + 1);
	}
	pprint_mat(img, width, height, channels);
	im2col(img, dst, width, height, channels, out_w, out_h, ksize, stride, pad);
	float* dst0 = dst;
	dst += dst_w * dst_h;
	dst[0] = -1;
	pprint_mat(dst0, dst_w, dst_h, 2);
	

	/*int n_filters = 1;
	int Awidth = ksize * ksize * channels;
	int Aheight = n_filters;
	int Asize = Awidth * Aheight;
	float* A = (float*)xcalloc(Asize, sizeof(float));
	for (int i = 0; i < Asize; i++) {
		A[i] = ((float)i) * (0.1f);
	}
	pprint_mat(A, Awidth, Aheight, 1);
	float* C = (float*)xcalloc(n_filters * dst_w, sizeof(float));
	gemm(n_filters, dst_w, Awidth, A, dst, C);
	pprint_mat(C, out_w, out_h, n_filters);*/
}

// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
/*height and width are of data_im.*/
void col2im_cpu(const float* data_col,
	int channels, int height, int width,
	int kernel_size, int pad, int stride,
	float* data_im)
{
	int output_h = (height + 2 * pad - kernel_size) / stride + 1;
	int output_w = (width + 2 * pad - kernel_size) / stride + 1;
	int channel_size = height * width;
	int channel, kernel_row, kernel_col, output_rows, output_col;
	for (channel = channels; channel--; data_im += channel_size) {
		for (kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
			for (kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
				int input_row = -pad + kernel_row;
				for (output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
						data_col += output_w;
					}
					else {
						int input_col = -pad + kernel_col;
						for (output_col = output_w; output_col; output_col--) {
							if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
								data_im[input_row * width + input_col] += *data_col;
							}
							data_col++;
							input_col += stride;
						}
					}
					input_row += stride;
				}
			}
		}
	}
}

void col2im(float* data_col, float* data_im,
	int w, int h, int channels,
	int out_w, int out_h,
	int ksize, int stride, int pad)
{
	int out_wh = out_w * out_h;
	int im_wh = w * h;
	int ch;
#pragma omp parallel for firstprivate(w, h, out_w, out_h, ksize, pad, stride)
	for (ch = 0; ch < channels; ch++) {
		float* im = &data_im[ch * im_wh];
		float* cm = &data_col[ch * out_wh * ksize * ksize];
		for (int krow = 0; krow < ksize; krow++) {
			for (int kcol = 0; kcol < ksize; kcol++) {
				int im_row = krow - pad;
				for (int out_rows = out_h; out_rows; out_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(im_row, h)) {
						cm += out_w;
					}
					else {
						int im_col = kcol - pad;
						for (int out_col = out_w; out_col; out_col--) {
							if (is_a_ge_zero_and_a_lt_b(im_col, w)) {
								im[im_row * w + im_col] += *cm;
							}
							cm++;
							im_col += stride;
						}
					}
					im_row += stride;
				}
			}
		}
	}
}

void test_col2im(void) {
	int width = 3;
	int height = 3;
	int channels = 1;
	int im_size = width * height * channels;
	float* data_im = (float*)xcalloc(im_size, sizeof(float));
	int pad = 0;
	int stride = 1;
	int ksize = 2;
	int out_size = (width + 2 * pad - ksize) / stride + 1; // square image
	int n = ksize * ksize * channels * out_size * out_size;
	float* data_col = (float*)xcalloc((size_t)n, sizeof(float));
	for (int i = 0; i < n; i++) {
		data_col[i] = 1.0F;
	}
	pprint_mat(data_col, out_size * out_size, ksize * ksize * channels, 1);
	col2im_cpu(data_col, channels, height, width, ksize, pad, stride, data_im);
	pprint_mat(data_im, width, height, channels);
}



// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
void im2col_cpu_general(const float* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	float* data_col)
{
	const int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	const int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	const int channel_size = height * width;
	int channel, kernel_row, kernel_col, output_rows, output_col;
	for (channel = channels; channel--; data_im += channel_size) {
		for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
				int input_row = -pad_h + kernel_row;
				for (output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
						for (output_col = output_w; output_col; output_col--) {
							*(data_col++) = 0;
						}
					}
					else {
						int input_col = -pad_w + kernel_col;
						for (output_col = output_w; output_col; output_col--) {
							if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
								*(data_col++) = data_im[input_row * width + input_col];
							}
							else {
								*(data_col++) = 0;
							}
							input_col += stride_w;
						}
					}
					input_row += stride_h;
				}
			}
		}
	}
}

// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
void col2im_cpu_general(const float* data_col, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	float* data_im)
{
	//caffe_set(height * width * channels, 0.0F, data_im);
	const int output_h = (height + 2 * pad_h -
		(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width + 2 * pad_w -
		(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	const int channel_size = height * width;
	int channel, kernel_row, kernel_col, output_rows, output_col;
	for (channel = channels; channel--; data_im += channel_size) {
		for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
				int input_row = -pad_h + kernel_row * dilation_h;
				for (output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
						data_col += output_w;
					}
					else {
						int input_col = -pad_w + kernel_col * dilation_w;
						for (output_col = output_w; output_col; output_col--) {
							if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
								data_im[input_row * width + input_col] += *data_col;
							}
							data_col++;
							input_col += stride_w;
						}
					}
					input_row += stride_h;
				}
			}
		}
	}
}