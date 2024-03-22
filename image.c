#include "image.h"
#include "xallocs.h"



image new_image(size_t w, size_t h, size_t c) {
    //image* img = (image*)xcalloc(1, sizeof(image));
    image img = { 0 };
    img.w = w;
    img.h = h;
    img.c = c;
    img.data = (float*)xcalloc(h * w * c, sizeof(float));
    return img;
}

void free_image(image* img) {
    xfree(img->data);
    xfree(img);
}

void print_image_matrix(image* im) {
    image img = *im;
    int i, j, k;
    printf("\n%zu X %zu Image:\n", img.w, img.h);
    for (k = 0; k < img.c; ++k) {
        if (k == 0) {
            printf("BLUE\n");
        }
        else if (k == 1) {
            printf("GREEN\n");
        }
        else {
            printf("RED\n");
        }
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