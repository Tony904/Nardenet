#include <stdio.h>
#include <math.h>
#include "xcuda.h"
#include "utils.h"


#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#define __shfl_down_sync(...) ( __VA_ARGS__ )
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif


#define NTSC_RED 0.299F
#define NTSC_GREEN 0.587
#define NTSC_BLUE 0.114


__global__ void resize_image_bilinear_gpu(float* input, float* output, int w, int h, int c, int out_w, int out_h, int out_c) {

    int tid = threadIdx.x;
    int wh = w * h;
    int out_wh = out_w * out_h;

    // need to subtract by 1 or else pixels won't map correctly between input and output images
    float w_ratio = (float)(w - 1) / (float)(out_w - 1);
    float h_ratio = (float)(h - 1) / (float)(out_h - 1);

    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            float px = x * w_ratio;
            float py = y * h_ratio;
            int x1 = (int)px;
            int x2 = x1 + 1;
            if (x2 >= w) x2 = w - 1;
            int y1 = (int)py;
            int y2 = py + 1;
            if (y2 >= h) y2 = h - 1;
            float dx = x1 ? px / (float)x1 : 0.0F;
            float dy = y1 ? py / (float)y1 : 0.0F;

            for (int ch = 0; ch < c; ch++) {
                float q1 = input[ch * wh + y1 * w + x1];
                float q2 = input[ch * wh + y1 * w + x2];
                float q3 = input[ch * wh + y2 * w + x1];
                float q4 = input[ch * wh + y2 * w + x2];

                float q12 = (q2 - q1) * dx + q1;
                float q34 = (q4 - q3) * dx + q3;

                output[ch * out_w * out_h + y * out_w + x] = (q34 - q12) * dy + q12;
            }
        }
    }
}