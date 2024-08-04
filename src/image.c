#include "image.h"
#include "xallocs.h"
#include "stdio.h"
#include "stbimage.h"
#include "utils.h"

#ifdef OPENCV
#include "xopencv.h"
#endif

#pragma warning(disable:4100)  // 'img' unreferenced formal parameter (when OPENCV is not defined)
/* REQUIRES COMPILATION WITH OPENCV */
void show_image(image* img) {
#ifdef OPENCV
    show_image_opencv(img->data, (int)img->w, (int)img->h, (int)img->c, 0);
#else
    printf("Cannot display image. Nardenet must be compiled with OpenCV installed "
           "and the preprocessor symbol OPENCV defined.\nPress any key to continue.\n");
    (void)getchar();
#endif
}

void load_image_to_buffer(char* filename, image* dst) {
    //img->data = load_image_opencv(filename, &img->w, &img->h, &img->c);
    load_image_stbi_to_buffer(filename, &dst->w, &dst->h, &dst->c, dst->data);
}

// returned img dimensions are absolute
image* load_image(char* filename) {
    image* img = (image*)xcalloc(1, sizeof(image));
    img->data = load_image_stbi(filename, &img->w, &img->h, &img->c);
    return img;
}

void write_image(image* img, char* filename) {
    write_image_stbi(filename, img->data, (int)img->w, (int)img->h, (int)img->c);
}

void write_image_test(void) {
    int w = 2;
    int h = 2;
    int c = 3;
    char* filename = "D:\\TonyDev\\NardeNet\\data\\testimg.png";
    //float* data = (float*)xcalloc((size_t)(w * h * c), sizeof(float));
    //fill_array_increment(data, (size_t)(w * h * c), 0.0F, 21.0F);
    float data[12] = { 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0 };
    pprint_mat(data, w, h, c);
    write_image_stbi(filename, data, w, h, c);
}

image* new_image(size_t w, size_t h, size_t c) {
    image* img = (image*)xcalloc(1, sizeof(image));
    img->w = w;
    img->h = h;
    img->c = c;
    img->data = (float*)xcalloc(h * w * c, sizeof(float));
    return img;
}

void free_image(image* img) {
    xfree(img->data);
    xfree(img);
}

void print_image_matrix(image* im) {
    if (!im) return;
    image img = *im;
    int i, j, k;
    printf("\nimage address: %p\n", im);
    printf("\n%zu X %zu Image:\n", img.w, img.h);
    for (k = 0; k < img.c; ++k) {
        if (k == 0) printf("BLUE\n");
        else if (k == 1) printf("GREEN\n");
        else printf("RED\n");
        printf(" __");
        for (j = 0; j < 4 * img.w - 1; ++j) printf(" ");
        printf("__ \n");
        printf("|  ");
        for (j = 0; j < 4 * img.w - 1; ++j) printf(" ");
        printf("  |\n");
        for (i = 0; i < img.h; ++i) {
            printf("|  ");
            for (j = 0; j < img.w; ++j) {
                printf("%0.1f ", img.data[k * img.w * img.h + i * img.w + j]); //im.data[ci * w * h + hi * w + wi]
            }
            printf(" |\n");
        }
        printf("|__");
        for (j = 0; j < 4 * img.w - 1; ++j) printf(" ");
        printf("__|\n");
    }
}

LIB_API int square(int x) {
    int y = x * 2;
    return y;
}