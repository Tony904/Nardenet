#ifndef IMAGE_H
#define IMAGE_H

#include "lib_api.h"
#include "stdlib.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
    
    typedef struct image image;
    
    image* load_image(char* filename);
    void load_image_to_buffer(char* filename, image* dst, int resize);
    void show_image(image* img);
    void write_image(image* img, char* filename);
    void write_image_test(void);
    void load_image_test(void);
    image* new_image(size_t width, size_t height, size_t channels);
    void normalize_image(image* img);
    void unnormalize_image(image* img);
    void resize_image_bilinear(image* dst, image* src);
    void randomize_colorspace(image* img, float brightness_lower, float brightness_upper, float contrast_lower, float contrast_upper, float saturation_lower, float saturation_upper, float hue_lower, float hue_upper);
    void transform_colorspace(image* img, float brightness_multi, float contrast_multi, float saturation_multi, float hue_shift);
    void free_image(image* img);
    void print_image_matrix(image* im);

    typedef struct image {
        size_t w;
        size_t h;
        size_t c;
        float* data;
    } image;

#ifdef __cplusplus
}
#endif
#endif