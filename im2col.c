#include "im2col.h"
#include <stdlib.h>
#include "xallocs.h"
#include "gemm.h"


void test_im2col(void);
void pprint_mat(float* data, int width, int height, int channels);


inline static int is_a_ge_zero_and_a_lt_b(int a, int b) {
	return (unsigned)(a) < (unsigned)(b);
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

void im2col_cpu(const float* data_im, const int channels,
	const int height, const int width, const int ksize,
	const int pad, const int stride,
	float* data_col)
{
	const int output_h = (height + 2 * pad - ksize) / stride + 1;
	const int output_w = (width + 2 * pad - ksize) / stride + 1;
	const int channel_size = height * width;
	int channel, kernel_row, kernel_col, output_rows, output_col;
	for (channel = channels; channel--; data_im += channel_size) {
		for (kernel_row = 0; kernel_row < ksize; kernel_row++) {
			for (kernel_col = 0; kernel_col < ksize; kernel_col++) {
				int input_row = -pad + kernel_row;
				for (output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
						for (output_col = output_w; output_col; output_col--) {
							*(data_col++) = 0;
						}
					}
					else {
						int input_col = -pad + kernel_col;
						for (output_col = output_w; output_col; output_col--) {
							if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
								*(data_col++) = data_im[input_row * width + input_col];
							}
							else {
								*(data_col++) = 0;
							}
							input_col += stride;
						}
					}
					input_row += stride;
				}
			}
		}
	}
}

void test_im2col(void) {
	int width = 3;
	int height = 3;
	int channels = 3;
	int img_size = width * height * channels;
	float* img = (float*)xcalloc(img_size, sizeof(float));
	int pad = 1;
	int stride = 1;
	int ksize = 3;
	int out_w = (width + pad * 2 - ksize) / stride + 1;
	int out_h = (height + pad * 2 - ksize) / stride + 1;
	int dst_w = out_w * out_h;
	int dst_h = ksize * ksize * channels;
	int dst_size = dst_w * dst_h;
	float* dst = (float*)xcalloc(dst_size, sizeof(float));

	for (int i = 0; i < img_size; i++) {
		img[i] = (float)(i + 1);
	}
	pprint_mat(img, width, height, channels);
	im2col_cpu(img, channels, height, width, ksize, pad, stride, dst);
	pprint_mat(dst, dst_w, dst_h, 1);

	int n_filters = 1;
	int Awidth = ksize * ksize * channels;
	int Aheight = n_filters;
	int Asize = Awidth * Aheight;
	float* A = (float*)xcalloc(Asize, sizeof(float));
	for (int i = 0; i < Asize; i++) {
		A[i] = ((float)i) * (0.1f);
	}
	pprint_mat(A, Awidth, Aheight, 1);
	float* C = (float*)xcalloc(n_filters * dst_w, sizeof(float));
	mmm_v2(n_filters, dst_w, Awidth, A, dst, C);
	pprint_mat(C, out_w, out_h, n_filters);
}

void pprint_mat(float* data, int width, int height, int channels) {
	printf("\nMATRIX");
	for (int channel = 0; channel < channels; channel++) {
		for (int row = 0; row < height; row++) {
			printf("\n");
			for (int col = 0; col < width; col++) {
				float val = data[channel * width * height + row * width + col];
				if (val < 10 && val >= 0) printf("%0.1f   ", val);
				else if (val >= 10 && val < 100) printf("%0.1f  ", val);
				else printf("%0.1f ", val);
			}
		}
		printf("(ch%d)", channel);
	}
	printf("\nend\n\n");
}
