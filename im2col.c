#include "im2col.h"
#include <stdlib.h>
#include "xallocs.h"


void test_img2col(void);
void pprint_mat(float* data, int width, int height);


void img2col(float* img, int width, int height, int channels, 
	 int ksize, int pad, int stride, 
	 float* dst)
{
	//float* dst = (float*)xcalloc(
	//l->out_w = ((l->w + (l->pad * 2) - l->ksize) / l->stride) + 1;

	int out_w = (width + pad * 2 - ksize) / stride + 1;
	int out_h = (height + pad * 2 - ksize) / stride + 1;
	int dst_w = out_w * out_h;
	//int dst_h = ksize * ksize * channels;
	int ky, kx, c;  // kernel row index, kernel col index, img channel index
	int dst_x, dst_y;
	float pixel;
	printf("\nout_w = %d, out_h = %d, dst_w = %d\n", out_w, out_h, dst_w);
	printf("\nIMG2COL\n");
	for (c = 0; c < channels; c++) {
		for (ky = 0; ky < ksize; ky++) {
			//printf("--");
			for (kx = 0; kx < ksize; kx++) {
				printf("\n");
				for (int img_y = ky - pad; img_y + (ksize - ky - 1) < out_h * stride; img_y += stride) {
					for (int img_x = kx - pad; img_x + (ksize - kx - 1) < out_w * stride; img_x += stride) {
						dst_y = c * ksize * ksize + ky * ksize + kx;
						int out_y = (img_y + ksize - ky - 1) / stride;
						dst_x = (out_w * out_y) + (img_x + ksize - kx - 1) / stride;
						if ((img_y < 0) || (img_x < 0)) { 
							pixel = 0;
						}
						else if ((img_y >= height) || (img_x >= width)) {
							pixel = 0;
						}
						else {
							pixel = img[img_y * width + img_x];
						}
						dst[dst_y * dst_w + dst_x] = pixel;
						if (pixel < 10 && pixel >= 0) printf("  %0.0f ", pixel);
						else if (pixel < 0) printf(" %0.0f ", pixel);
						else printf("%0.0f ", pixel);
						printf(" out_w = %d out_y = %d ", out_w, out_y);
						printf(" math = %d ", (img_x + ksize - kx - 1) / stride);
						printf(" dst_y = %d, dst_x = %d, dst_y * dst_w + dst_x = %d\n", dst_y, dst_x, dst_y * dst_w + dst_x);
					}
					//printf(" -\n");
				}
			}
		}
	}
	printf("\nEND\n\n");
}

void test_img2col(void) {
	int width = 3;
	int height = 3;
	int channels = 1;
	int img_size = width * height * channels;
	float* img = (float*)xcalloc(img_size, sizeof(float));
	int pad = 1;
	int stride = 1;
	int ksize = 2;
	int out_w = (width + pad * 2 - ksize) / stride + 1;
	int out_h = (height + pad * 2 - ksize) / stride + 1;
	int dst_w = out_w * out_h;
	int dst_h = ksize * ksize * channels;
	int dst_size = dst_w * dst_h;
	float* dst = (float*)xcalloc(dst_size, sizeof(float));

	for (int i = 0; i < img_size; i++) {
		img[i] = (float)(i + 1);
	}
	pprint_mat(img, width, height);
	img2col(img, width, height, channels, ksize, pad, stride, dst);
	for (int i = 0; i < dst_size; i++) {
		printf("%0.0f ", dst[i]);
	}
	printf("\n");
	//pprint_mat(dst, dst_w, dst_h);

}

void pprint_mat(float* data, int width, int height) {
	printf("\nMATRIX\n");
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			float val = data[row * width + col];
			if (val < 10 && val >= 0) printf("  %0.0f ", val);
			else if (val < 0) printf(" %0.0f ", val);
			else printf("%0.0f ", val);
		}
		printf("\n");
	}
	printf("end\n\n");
}
