#include "image.h"
#include "xallocs.h"
#include "stdio.h"
#include "stbimage.h"
#include "utils.h"

#ifdef OPENCV
#include "xopencv.h"
#endif


#define NTSC_RED 0.299F
#define NTSC_GREEN 0.587
#define NTSC_BLUE 0.114


void rgb2hsv(image* img);


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

/* resize = 1 means the image will be resized to fit the dimensions of dst */
void load_image_to_buffer(char* filename, image* dst, int resize) {
    //img->data = load_image_opencv(filename, &img->w, &img->h, &img->c);
    if (!resize) {
        load_image_stbi_to_buffer(filename, &dst->w, &dst->h, &dst->c, dst->data);
        return;
    }
    image img = { 0 };
    img.data = load_image_stbi(filename, &img.w, &img.h, &img.c);
    resize_image_bilinear(dst, &img);
    xfree(img.data);
}

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
    float data[12] = { 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0 };  // should create img with the red channel filled with 255
    pprint_mat(data, w, h, c);
    write_image_stbi(filename, data, w, h, c);
}

void load_image_test(void) {
    char* filename = "D:\\TonyDev\\NardeNet\\data\\testimg.png";
    image* img = load_image(filename);
    pprint_mat(img->data, (int)img->w, (int)img->h, (int)img->c);
}

image* new_image(size_t w, size_t h, size_t c) {
    image* img = (image*)xcalloc(1, sizeof(image));
    img->w = w;
    img->h = h;
    img->c = c;
    img->data = (float*)xcalloc(h * w * c, sizeof(float));
    return img;
}

/* Set w and h of dst to resized dimensions before passing.
   Note: dst.c will be set equal to src.c */
void resize_image_bilinear(image* dst, image* src) {
    size_t dst_w = dst->w;
    size_t dst_h = dst->h;
    size_t src_w = src->w;
    size_t src_h = src->h;
    size_t c = src->c;
    dst->c = c;

    if (!dst_w || !dst_h) {
        printf("Cannot resize an image dimension to zero.\n");
        wait_for_key_then_exit();
    }

    size_t src_wh = src_w * src_h;
    size_t dst_wh = dst_w * dst_h;

    float* src_data = src->data;
    float* dst_data = dst->data;

    if (dst_w == src_w && dst_h == src_h) {
        size_t n = dst_wh * c;
        for (size_t i = 0; i < n; i++) {
            dst_data[i] = src_data[i];
        }
        return;
    }

    // need to subtract by 1 or else pixels won't map correctly between input and output images
    float w_ratio = (float)(src_w - 1) / (float)(dst_w - 1);
    float h_ratio = (float)(src_h - 1) / (float)(dst_h - 1);

    for (size_t y = 0; y < dst_h; y++) {
        for (size_t x = 0; x < dst_w; x++) {
            float px = x * w_ratio;
            float py = y * h_ratio;
            size_t x1 = (size_t)px;
            size_t x2 = x1 + 1;
            if (x2 >= src_w) x2 = src_w - 1;
            size_t y1 = (size_t)py;
            size_t y2 = py + 1;
            if (y2 >= src_h) y2 = src_h - 1;
            float dx = x1 ? px / (float)x1 : 0.0F;
            float dy = y1 ? py / (float)y1 : 0.0F;

            for (size_t ch = 0; ch < c; ch++) {
                float q1 = src_data[ch * src_wh + y1 * src_w + x1];
                float q2 = src_data[ch * src_wh + y1 * src_w + x2];
                float q3 = src_data[ch * src_wh + y2 * src_w + x1];
                float q4 = src_data[ch * src_wh + y2 * src_w + x2];

                float q12 = (q2 - q1) * dx + q1;
                float q34 = (q4 - q3) * dx + q3;

                dst_data[ch * dst_wh + y * dst_w + x] = (q34 - q12) * dy + q12;
            }
        }
    }
}

void change_brightness(image* img, float multiplier) {
    float* data = img->data;
    size_t n = img->w * img->h * img->c;
    for (size_t i = 0; i < n; i++) {
        data[i] *= multiplier;
        if (data[i] > 255.0F) data[i] = 255.0F;
    }
}

/* https://www.cs.rit.edu/~ncs/color/t_convert.html */
void rgb2hsv(image* img) {
    size_t w = img->w;
    size_t h = img->h;
    size_t c = img->c;
    float* data = img->data;
    size_t wh = w * h;
    for (size_t s = 0; s < wh; s++) {
        float red = data[s];
        float green = data[wh + s];
        float blue = data[wh * 2 + s];

        float min_val;
        float max_val;

        if (red < green) min_val = (red < blue) ? red : blue;
        else min_val = (green < blue) ? green : blue;

        if (red > green) max_val = (red > blue) ? red : blue;
        else max_val = (green > blue) ? green : blue;

        float V = max_val;
        float delta = max_val - min_val;
        float S = 0.0F;
        float H = -1.0F;
        if (max_val) S = delta / max_val;
        else return;

        if (red == max_val) H = (green - blue) / delta;
        else if (green == max_val) H = 2.0F + (blue - red) / delta;
        else H = 4.0F + (red - green) / delta;

        H *= 60.0F;
        if (H < 0.0F) H += 360.0F;

        data[s] = H;
        data[wh + s] = S;
        data[wh * 2] = V;
    }
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