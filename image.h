#ifndef IMAGE_H
#define IMAGE_H



#ifdef __cplusplus
extern "C" {
#endif

    
    //typedef struct image image;
    typedef struct image {
        size_t w;
        size_t h;
        size_t c;
        float* data;
    } image;

    image new_image(size_t width, size_t height, size_t channels);
    void free_image(image* img);
    void print_image_matrix(image* im);

    


#ifdef __cplusplus
}
#endif
#endif