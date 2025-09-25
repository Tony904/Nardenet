#ifdef GPU

#include <stdio.h>
#include "xcuda.h"
#include "xallocs.h"
#include "utils.h"
#include "im2col.h"
#include <math.h>
#include "blas.h"


#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#define __shfl_down_sync(...) ( __VA_ARGS__ )
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif


#ifndef min
#define min(x, y) ((x > y) ? y : x)
#endif


/*******************************************
                   IM2COL
*******************************************/

__global__ void im2col_kernel(const float* __restrict__ data_im,
    const int width, const int height, const int channels,
    const int ksize, const int pad, const int stride,
    const int width_out, const int height_out,
    float* __restrict__ data_col,
    int n) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
        const int col_out = index % width_out;
        const int row_out = (index / width_out) % height_out;
        const int ch_in = (index / width_out / height_out) % channels;
        const int row_in = row_out * stride - pad;
        const int col_in = col_out * stride - pad;
        const int data_im_offset = (ch_in * height + row_in) * width + col_in;
        const int data_col_offset = ((ch_in * ksize * ksize) * height_out * width_out) + (row_out * width_out + col_out);
        for (int krow = 0; krow < ksize; ++krow) {
            for (int kcol = 0; kcol < ksize; ++kcol) {
                const int row = row_in + krow;
                const int col = col_in + kcol;
                const int data_col_index = data_col_offset + ((krow * ksize + kcol) * height_out * width_out);
                if (row >= 0 && row < height && col >= 0 && col < width) {
                    data_col[data_col_index] = data_im[data_im_offset + krow * width + kcol];
                }
                else data_col[data_col_index] = 0.0F;
            }
        }
    }
}
void im2col_gpu(float* data_im, float* data_col, int im_w, int im_h, int im_c, int out_w, int out_h, int ksize, int stride, int pad) {
    int n = out_w * out_h * im_c;
    int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
    im2col_kernel KARGS(grid_size, BLOCKSIZE) (data_im, im_w, im_h, im_c, ksize, pad, stride, out_w, out_h, data_col, n);
    CHECK_CUDA(cudaPeekAtLastError());
}

// leaving this here for future me as a reminder that this is not faster
__global__ void im2col_kernel_that_uses_shared_memory_but_is_somehow_slower(const float* __restrict__ im, float* __restrict__ ex,
    const int im_width, const int im_height, const int im_channels,
    const int out_width, const int out_height,
    const int ksize, const int stride, const int pad) {

    extern __shared__ float shared[];

    int block_start_im_row = blockIdx.y * blockDim.y;
    int block_start_im_col = blockIdx.x * blockDim.x;
    int shared_width = blockDim.x + ksize - 1;
    int shared_height = blockDim.y + ksize - 1;

    int k = (ksize - 1) / 2;

    for (int row = threadIdx.y; row < shared_height; row+=blockDim.y) {
        for (int col = threadIdx.x; col < shared_width; col+=blockDim.x) {
            int im_row = block_start_im_row + row - k;
            int im_col = block_start_im_col + col - k;
            if (im_row >= 0 && im_col >= 0 && im_row < im_height && im_col < im_width) {
                shared[row * shared_width + col] = im[blockIdx.z * im_width * im_height + im_row * im_width + im_col];
            }
            else shared[row * shared_width + col] = 0.0F;
        }
    }
    __syncthreads();

    int thread_im_row = block_start_im_row + threadIdx.y;
    int thread_im_col = block_start_im_col + threadIdx.x;
    
    int x0 = k - pad;  // 0th stride row of kernel center
    int y0 = x0;  // y0 is same as x0 since kernel, padding, and input are squares
    if (thread_im_row + k < im_height + pad && thread_im_col + k < im_width + pad && thread_im_row >= y0 && thread_im_col >= x0) {  // valid range check
        if (((thread_im_row - y0) % stride) == 0 && ((thread_im_col - x0) % stride) == 0) {  // if position corresponds to center of a kernel stride
            int out_row = (thread_im_row - y0) / stride;
            int out_col = (thread_im_col - x0) / stride;
            for (int krow = 0; krow < ksize; krow++) {
                for (int kcol = 0; kcol < ksize; kcol++) {
                    int shared_index = (threadIdx.y + krow) * shared_width + threadIdx.x + kcol;
                    int ex_index = ((blockIdx.z * ksize * ksize) + (krow * ksize) + kcol) * (out_width * out_height) + (out_row * out_width) + out_col;
                    ex[ex_index] = shared[shared_index];
                }
            }
        }
    }
}
void im2col_gpu_shared_memory_and_slower(float* data_im, float* data_col, int channels, int h, int w, int ksize, int stride, int pad, int out_h, int out_w) {
    dim3 block_size(16, 16);
    dim3 grid_size((out_w + block_size.x - 1) / block_size.x, (out_h + block_size.y - 1) / block_size.y, channels);
    size_t shared_memory_size = (block_size.x + 2 * pad) * (block_size.y + 2 * pad) * sizeof(float);
    im2col_kernel_that_uses_shared_memory_but_is_somehow_slower KARGS(grid_size, block_size, shared_memory_size) (data_im, data_col, channels, h, w, ksize, stride, pad, out_h, out_w);
    CHECK_CUDA(cudaPeekAtLastError());
}

void cuda_test_im2col(void) {
    int width = 123;
    int height = width;
    int channels = 32;
    int pad = 0;
    int stride = 3;
    int ksize = 5;
    if (ksize % 2 == 0) {
        printf("ksize must be even, is %d\n", ksize);
        wait_for_key_then_exit();
    }

    int im_n = width * height * channels;
    float* im = (float*)xmalloc(im_n * sizeof(float));
    
    int out_w = (width + pad * 2 - ksize) / stride + 1;
    int out_h = (height + pad * 2 - ksize) / stride + 1;

    int dst_w = out_w * out_h;
    int dst_h = ksize * ksize * channels;
    int dst_n = dst_w * dst_h;
    float* col = (float*)xcalloc(dst_n, sizeof(float));

    //fill_array_rand_float(im, im_n, 0., 1.);
    fill_array_increment(im, im_n, 1., 1.);

    float* d_im = 0;
    float* d_col = 0;

    CHECK_CUDA(cudaMalloc(&d_im, im_n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_col, dst_n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_im, im, sizeof(float) * im_n, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    im2col_gpu(d_im, d_col, width, height, channels, out_w, out_h , ksize, stride, pad);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(col, d_col, sizeof(float) * dst_n, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_col));
    CHECK_CUDA(cudaFree(d_im));

    float* col_cpu = (float*)xcalloc(dst_n, sizeof(float));
    im2col(im, col_cpu, width, height, channels, out_w, out_h, ksize, stride, pad);

    printf("Verifiying......\n");
    float epsilon = 1e-5f;
    size_t zero_count = 0;
    for (size_t i = 0; i < dst_n; i++) {
        //printf("%f =? %f\n", col_cpu[i], col[i]);
        if (fabs(col_cpu[i] - col[i]) > epsilon || isnan(col_cpu[i]) || isnan(col[i])) {
            printf("Verification Failed: i = %zu, (col_cpu)%f != (col_gpu)%f\n", i, col_cpu[i], col[i]);
            wait_for_key_then_exit();
        }
        if (col_cpu[i] == 0.0F && col[i] == 0.0F) {
            zero_count++;
            printf("zero count: %zu\r", zero_count);
        }
    }
    printf("zero count: %zu\n", zero_count);
    printf("Verifiction Success!!!\n");
}



/*******************************************
                   COL2IM
*******************************************/

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE
__global__ void col2im_kernel(const float* __restrict__ data_col,
    const int out_w, const int out_h,
    const int ksize, const int pad, const int stride,
    const int width, const int height,
    float* __restrict__ data_im,
    const int n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < n; index += blockDim.x * gridDim.x) {
        float val = 0;
        int w = index % width + pad;
        int h = (index / width) % height + pad;
        int c = index / (width * height);
        int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
        int w_col_end = min(w / stride + 1, out_w);
        int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
        int h_col_end = min(h / stride + 1, out_h);
        int offset = (c * ksize * ksize + h * ksize + w) * out_h * out_w;
        int coeff_h_col = (1 - stride * ksize * out_h) * out_w;
        int coeff_w_col = (1 - stride * out_h * out_w);
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
            }
        }
        data_im[index] += val;
    }
}
void col2im_gpu(float* data_col, float* data_im, int im_width, int im_height, int out_w, int out_h, int ksize, int stride, int pad, int n) {
    int grid_size = GET_GRIDSIZE(n, BLOCKSIZE);
    col2im_kernel KARGS(grid_size, BLOCKSIZE) (data_col, out_w, out_h, ksize, pad, stride, im_width, im_height, data_im, n);
    CHECK_CUDA(cudaPeekAtLastError());
}

void cuda_test_col2im(void) {
    int width = 64;
    int height = width;
    int channels = 16;
    if (width % 32 != 0) {
        printf("Input width must be a multiple of 32.\n");
        exit(EXIT_FAILURE);
    }

    size_t pad = 1;
    size_t stride = 1;
    size_t ksize = 3;
    size_t out_w = (width + 2 * pad - ksize) / stride + 1;
    size_t out_h = out_w; // square image
    size_t col_n = ksize * ksize * channels * out_w * out_h;
    float* col = (float*)xmalloc(col_n * sizeof(float));
    fill_array_rand_float(col, col_n, 0., 1.);

    int im_n = width * height * channels;
    float* im = (float*)xmalloc(im_n * sizeof(float));

    float* d_im = 0;
    float* d_col = 0;

    CHECK_CUDA(cudaMalloc(&d_im, im_n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_col, col_n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_col, col, sizeof(float) * col_n, cudaMemcpyHostToDevice));


#pragma warning (suppress:4267)
    col2im_gpu(d_col, d_im, height, width, out_w, out_h, ksize, stride, pad, im_n);


    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(im, d_im, sizeof(float) * im_n, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_col));
    CHECK_CUDA(cudaFree(d_im));

    //pprint_mat(col, dst_w, dst_h, 1);
    float* im_cpu = (float*)xcalloc(im_n, sizeof(float));
#pragma warning (suppress:4267)
    col2im(col, im_cpu, width, height, channels, out_w, out_h, ksize, stride, pad);

    float epsilon = 1e-5f;
    size_t zero_count = 0;
    printf("Verifiying......\n");
    for (size_t i = 0; i < im_n; i++) {
        //printf("%f =? %f\n", im_cpu[i], im[i]);
        if (fabs(im_cpu[i] - im[i]) > epsilon) {
            printf("Verification Failed: i = %zu, (im_cpu)%f != (im_gpu)%f\n", i, im_cpu[i], im[i]);
            wait_for_key_then_exit();
        }
        if (im_cpu[i] == 0.0F && im[i] == 0.0F) {
            zero_count++;
            printf("zero count: %zu\r", zero_count);
        }
    }
    printf("zero count: %zu\n", zero_count);
    printf("Verifiction Success!!!\n");
}

#endif