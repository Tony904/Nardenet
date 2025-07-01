#include <stdio.h>
#include <math.h>
#include "xcuda.h"
#include "image.h"


#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#define __shfl_down_sync(...) ( __VA_ARGS__ )
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif


__global__ void scale_contrast_rgb_kernel(float* data, int spatial, float scalar) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < spatial) {
        float red = data[i];
        float green = data[spatial + i];
        float blue = data[spatial * 2 + i];

        float mean = (red + green + blue) / 3.0F;

        red = mean + (red - mean) * scalar;
        green = mean + (green - mean) * scalar;
        blue = mean + (blue - mean) * scalar;

        data[i] = red;
        data[spatial + i] = green;
        data[spatial * 2 + i] = blue;
    }
}
void scale_contrast_rgb_gpu(image* img, float scalar) {
    int spatial = (int)(img->w * img->h);
    int grid_size = GET_GRIDSIZE(spatial, BLOCKSIZE);
    scale_contrast_rgb_kernel KARGS(grid_size, BLOCKSIZE) (img->data, spatial, scalar);
    CHECK_CUDA(cudaPeekAtLastError());
}



__global__ void rgb2hsv_kernel(float* data, int spatial) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < spatial) {
        float red = data[i];
        float green = data[spatial + i];
        float blue = data[spatial * 2 + i];

        float min_val;
        float max_val;

        if (red < green) min_val = (red < blue) ? red : blue;
        else min_val = (green < blue) ? green : blue;

        if (red > green) max_val = (red > blue) ? red : blue;
        else max_val = (green > blue) ? green : blue;

        float V = max_val;
        float delta = max_val - min_val;
        float S = 0.0F;
        float H = 0.0F;
        if (max_val) {
            S = delta / max_val;

            if (red == max_val) H = (green - blue) / delta;
            else if (green == max_val) H = 2.0F + (blue - red) / delta;
            else H = 4.0F + (red - green) / delta;

            if (H < 0.0F) H += 6.0F;
            H /= 6.0F;
        }
        data[i] = H;
        data[spatial + i] = S;
        data[spatial * 2 + i] = V;
    }
}
void rgb2hsv_gpu(image* img) {
    int spatial = (int)(img->w * img->h);
    int grid_size = GET_GRIDSIZE(spatial, BLOCKSIZE);
    rgb2hsv_kernel KARGS(grid_size, BLOCKSIZE) (img->data, spatial);
    CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void hsv2rgb_kernel(float* data, int spatial) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < spatial) {
        float H = data[i] * 6.0F;
        float S = data[spatial + i];
        float V = data[spatial * 2 + i];

        if (S == 0.0F) { // if img is achromatic
            data[i] = data[spatial + i] = data[spatial * 2 + i] = V;
        }
        else {
            int n = (int)floorf(H);
            float f = H - (float)n;
            float p = V * (1.0F - S);
            float q = V * (1.0F - S * f);
            float t = V * (1.0F - S * (1.0F - f));
            float red;
            float green;
            float blue;
            switch (n) {
            case 0:
                red = V; green = t; blue = p;
                break;
            case 1:
                red = q; green = V; blue = p;
                break;
            case 2:
                red = p; green = V; blue = t;
                break;
            case 3:
                red = p; green = q; blue = V;
                break;
            case 4:
                red = t; green = p; blue = V;
                break;
            default:
                red = V; green = p; blue = q;
                break;
            }
            data[i] = red;
            data[spatial + i] = green;
            data[spatial * 2 + i] = blue;
        }
    }
}
void hsv2rgb_gpu(image* img) {
    int spatial = (int)(img->w * img->h);
    int grid_size = GET_GRIDSIZE(spatial, BLOCKSIZE);
    hsv2rgb_kernel KARGS(grid_size, BLOCKSIZE) (img->data, spatial);
    CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void hsv_apply_changes_kernel(float* data, int spatial, float brightness_scalar, float saturation_scalar, float hue_shift) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < spatial) {
        float val = data[i];
        val += hue_shift;
        if (val > 1.0F) val -= 1.0F;
        else if (val < 0.0F) val += 1.0F;
        data[i] = val;
        data[spatial + i] *= saturation_scalar;
        data[spatial * 2 + i] *= brightness_scalar;
    }
}
void hsv_apply_changes_gpu(image* img, float brightness_scalar, float saturation_scalar, float hue_shift) {
    int spatial = (int)(img->w * img->h);
    int grid_size = GET_GRIDSIZE(spatial, BLOCKSIZE);
    hsv_apply_changes_kernel KARGS(grid_size, BLOCKSIZE) (img->data, spatial, brightness_scalar, saturation_scalar, hue_shift);
    CHECK_CUDA(cudaPeekAtLastError());
}

void transform_colorspace_gpu(image* img, float brightness_scalar, float contrast_scalar, float saturation_scalar, float hue_shift) {
    if (contrast_scalar != 1.0F) scale_contrast_rgb_gpu(img, contrast_scalar);
    rgb2hsv_gpu(img);
    hsv_apply_changes_gpu(img, brightness_scalar, saturation_scalar, hue_shift);
    hsv2rgb_gpu(img);
}