#include "stbimage.h"
#include "xallocs.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "utils.h"
#include "xallocs.h"


// Basic usage:
//    int x,y,n;
//    unsigned char *data = stbi_load(filename, &x, &y, &n, 0);
//    // ... process data if not NULL ...
//    // ... x = width, y = height, n = # 8-bit components per pixel ...
//    // ... replace '0' with '1'..'4' to force that many components per pixel
//    // ... but 'n' will always be the number that it would have been if you said 0
//    stbi_image_free(data);
//
// Standard parameters:
//    int *x                 -- outputs image width in pixels
//    int *y                 -- outputs image height in pixels
//    int *channels_in_file  -- outputs # of image components in image file
//    int desired_channels   -- if non-zero, # of image components requested in result


void load_image_stbi_to_buffer(char* filename, size_t* w, size_t* h, size_t* c, float* dst) {
	// TODO: Add interpolation if buffer size does not match size read.
	int x;
	int y;
	int n;
	unsigned char* data = stbi_load(filename, &x, &y, &n, 0);
	if (!data) {
		printf("Failed to load image %s\n", filename);
		printf("Press enter to exit program.\n");
		(void)getchar();
		exit(EXIT_FAILURE);
	}
	size_t expected_size = (*w) * (*h) * (*c);
	size_t X = (size_t)x;
	size_t Y = (size_t)y;
	size_t N = (size_t)n;
	size_t size = X * Y * N;
	if (size != expected_size) {
		printf("Expected image size (%zu) does not equal actual image size (%zu).\n", expected_size, size);
		wait_for_key_then_exit();
	}
	for (size_t i = 0; i < size; i++) {
		dst[i] = (float)data[i];
	}
	stbi_image_free(data);
}

float* load_image_stbi(char* filename, size_t* w, size_t* h, size_t* c) {
	int x;
	int y;
	int n;
	unsigned char* data = stbi_load(filename, &x, &y, &n, 0);
	if (!data) {
		printf("Failed to load image %s\n", filename);
		printf("Press enter to exit program.\n");
		(void)getchar();
		exit(EXIT_FAILURE);
	}
	size_t X = (size_t)x;
	size_t Y = (size_t)y;
	size_t N = (size_t)n;
	*w = X;
	*h = Y;
	*c = N;
	size_t size = X * Y * N;
	float* dataf = (float*)xcalloc(size, sizeof(float));
	for (size_t i = 0; i < size; i++) {
		dataf[i] = (float)data[i];
	}
	stbi_image_free(data);
	return dataf;
}


void write_image_stbi(char* filename, float* data, int w, int h, int c) {
	int ext_i = get_filename_ext_index(filename);
	assert(ext_i > -1);
	char* ext = &filename[ext_i];
	unsigned char* ucdata = (unsigned char*)xcalloc((size_t)(w * h * c), sizeof(unsigned char));
	unsigned char* start = ucdata;
	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w; col++) {
			for (int ch = 0; ch < c; ch++) {
				int x = (int)data[row * w * c + col * c + ch];  // need to convert to int first before converting to unsigned char to get correct value
				*(ucdata++) = (unsigned char)x;
			}
		}
	}
	ucdata = start;
	if (strcmp(ext, ".png") == 0) stbi_write_png(filename, w, h, c, (void*)ucdata, 0); // idk if 0 for stride is correct
	else if (strcmp(ext, ".bmp") == 0) stbi_write_bmp(filename, w, h, c, (void*)ucdata);
	else if (strcmp(ext, ".jpg") == 0) stbi_write_jpg(filename, w, h, c, (void*)ucdata, 90);
	else {
		printf("Unsupported image format: %s\nPress enter to continue.", ext);
		(void)getchar();
	}
}