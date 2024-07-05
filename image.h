#ifndef IMAGE_H
#define IMAGE_H

#include "lib_api.h"
#include "stdlib.h"

#ifdef __cplusplus
extern "C" {
#endif
    
    typedef struct image image;
    
    void show_image(image* img);
    image* load_image(char* filename);
    image* new_image(size_t width, size_t height, size_t channels);
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