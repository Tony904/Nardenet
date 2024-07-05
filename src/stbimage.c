#include "stbimage.h"
#include "xallocs.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


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
		dataf[i] = (float)data[i] / 255.0F;
	}
	stbi_image_free(data);
	return dataf;
}