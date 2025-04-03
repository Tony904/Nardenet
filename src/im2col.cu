#include <stdio.h>
#include "xcuda.h"
#include "xallocs.h"
#include "utils.h"
#include "im2col.h"
#include <math.h>


#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif

#ifndef min
#define min(x, y) ((x > y) ? y : x)
#endif


// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

__global__ void col2im_kernel(const float* data_col,
    const int width_col, const int height_col,
    const int ksize, const int pad, const int stride,
    const int width, const int height,
    float* data_im,
    const int n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < n; index += blockDim.x * gridDim.x) {
        float val = 0;
        int w = index % width + pad;
        int h = (index / width) % height + pad;
        int c = index / (width * height);
        int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
        int w_col_end = min(w / stride + 1, width_col);
        int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
        int h_col_end = min(h / stride + 1, height_col);
        int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
        int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
        int coeff_w_col = (1 - stride * height_col * width_col);
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
            }
        }
        data_im[index] += val;
    }
}

void cuda_test_col2im(void) {
    int width = 320;
    int height = width;
    int channels = 80;
    if (width % 32 != 0) {
        printf("Input width must be a multiple of 32.\n");
        exit(EXIT_FAILURE);
    }

    int pad = 1;
    int stride = 1;
    int ksize = 3;
    int col_size = (width + 2 * pad - ksize) / stride + 1; // square image
    int col_n = ksize * ksize * channels * col_size * col_size;
    float* col = (float*)calloc((size_t)col_n, sizeof(float));
    if (!col) {
        fprintf(stderr, "Failed to calloc col.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < col_n; i++) {
        col[i] = 1.0F;
    }
    
    int im_n = width * height * channels;
    float* im = (float*)calloc(im_n, sizeof(float));
    if (!im) {
        fprintf(stderr, "Failed to calloc im.\n");
        exit(EXIT_FAILURE);
    }

    float* d_im = 0;
    float* d_col = 0;

    CHECK_CUDA(cudaMalloc(&d_im, im_n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_col, col_n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_col, col, sizeof(float) * col_n, cudaMemcpyHostToDevice));

    int threads_per_block = 512;
    int num_cuda_kernels = im_n;
    int num_blocks = (num_cuda_kernels + threads_per_block - 1) / threads_per_block;

    printf("num_blocks = %d\n", num_blocks);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    col2im_kernel KARGS(num_blocks, threads_per_block) (d_col, col_size, col_size,
        ksize, pad, stride,
        height, width,
        d_im,
        num_cuda_kernels);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("col2im kernel execution time: %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(im, d_im, sizeof(float) * im_n, cudaMemcpyDeviceToHost));

    //pprint_mat(col, dst_w, dst_h, 1);
    float* im_cpu = (float*)calloc(im_n, sizeof(float));
    if (!im_cpu) {
        fprintf(stderr, "Failed to calloc im_cpu.\n");
        exit(EXIT_FAILURE);
    }
    col2im(col, channels, height, width, ksize, pad, stride, im_cpu);

    printf("Verifiying......\n");
    for (size_t i = 0; i < im_n; i++) {
        //printf("%f =? %f\n", col_cpu[i], col[i]);
        if (im_cpu[i] != im[i]) {
            printf("Verification Failed: i = %zu, (im_cpu)%f != (im_gpu)%f\n", i, im_cpu[i], im[i]);
            wait_for_key_then_exit();
        }
    }
    printf("Verifiction Success!!!\n");
}


__global__ void im2col_kernel(const float* data_im, const int height, const int width, const int channels,
    const int ksize, const int pad, const int stride,
    const int height_out, const int width_out, float* data_col,
    int n) {

    //int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
        const int col_out = index % width_out;
        const int row_out = (index / width_out) % height_out;
        const int ch_in = (index / width_out / height_out) % channels;

        // Calculate corresponding top-left position in input
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
                else {
                    data_col[data_col_index] = 0.0F;
                }
            }
        }
    }
}

void cuda_test_im2col(void) {
    int width = 320;
    int height = 320;
    int channels = 80;
    if (width % 32 != 0) {
        printf("Input width must be a multiple of 32.\n");
        exit(EXIT_FAILURE);
    }
    int im_n = width * height * channels;
    float* im = (float*)calloc(im_n, sizeof(float));
    if (!im) {
        fprintf(stderr, "Failed to calloc im.");
        exit(EXIT_FAILURE);
    }
    int pad = 1;
    int stride = 1;
    int ksize = 3;
    
    int out_w = (width + pad * 2 - ksize) / stride + 1;
    int out_h = (height + pad * 2 - ksize) / stride + 1;

    int dst_w = out_w * out_h;
    int dst_h = ksize * ksize * channels;
    int dst_n = dst_w * dst_h;
    float* col = (float*)calloc(dst_n, sizeof(float));
    if (!col) {
        fprintf(stderr, "Failed to calloc col.\n");
        exit(EXIT_FAILURE);
    }

    fill_array_rand_float(im, im_n, 0, 1);

    float* d_im = 0;
    float* d_col = 0;

    CHECK_CUDA(cudaMalloc(&d_im, im_n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_col, dst_n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_im, im, sizeof(float) * im_n, cudaMemcpyHostToDevice));

    int threads_per_block = 512;
    int num_cuda_kernels = dst_n;
    int num_blocks = (num_cuda_kernels + threads_per_block - 1) / threads_per_block;

    printf("num_blocks = %d\n", num_blocks);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    im2col_kernel KARGS(num_blocks, threads_per_block) (d_im, height, width, channels,
        ksize, pad, stride,
        out_h, out_w, d_col,
        num_cuda_kernels);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time (no shared mem): %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(col, d_col, sizeof(float) * dst_n, cudaMemcpyDeviceToHost));

    //pprint_mat(col, dst_w, dst_h, 1);
    float* col_cpu = (float*)calloc(dst_n, sizeof(float));
    if (!col_cpu) {
        fprintf(stderr, "Failed to calloc col_cpu.\n");
        exit(EXIT_FAILURE);
    }
    im2col(im, channels, height, width, ksize, pad, stride, col_cpu);

    printf("Verifiying......\n");
    for (size_t i = 0; i < dst_n; i++) {
        //printf("%f =? %f\n", col_cpu[i], col[i]);
        if (col_cpu[i] != col[i]) {
            printf("Verification Failed: i = %zu, (col_cpu)%f != (col_gpu)%f\n", i, col_cpu[i], col[i]);
            wait_for_key_then_exit();
        }
    }
    printf("Verifiction Success!!!\n");
}

// Pad has to equal (ksize - 1) / 2. Block size must = input width. Stride must = 1.
__global__ void im2col_kernel_shared_custom(const float* data_im, const int height, const int width, const int channels,
    const int ksize,
    float* data_col) {

    extern __shared__ float shared_input[];     // Equals width_col * sizeof(float)

    // assert n == height_col * width_col * channels * ksize;
    int radius = 1;
    int share_row = blockIdx.y - radius;
    int share_col = threadIdx.x;
    int ch = blockIdx.z;

    if (share_row >= 0 && share_row < height) {
        shared_input[share_col] = data_im[(ch * height + share_row) * width + share_col];
    }
    __syncthreads();
    int start_row_out = share_row + radius;
    int start_col_in = share_col - radius;

    for (int krow = 0; krow < ksize; krow++) {
        int row_out = start_row_out - krow;
        if (row_out < 0 || row_out >= height) continue;
        int dst_index_base = (ch * ksize + krow) * ksize * height * width;
        for (int kcol = 0; kcol < ksize; kcol++) {

            int col_in = start_col_in + kcol;
            //int dst_index = (((ch * ksize + krow) * ksize + kcol) * height + row_out) * width + share_col;
            int dst_index = dst_index_base + (kcol * height + row_out) * width + share_col;
            if (col_in >= 0 && col_in < width && row_out >= 0 && row_out < height && share_row >= 0 && share_row < height) {
                data_col[dst_index] = shared_input[col_in];
            }
            else {
                data_col[dst_index] = 0.0F;
            }
        }
    }
}

void cuda_test_im2col_shared_custom(void) {
    int width = 320;
    int height = width;
    int channels = 512;

    int im_n = width * height * channels;
    float* im = (float*)malloc(im_n * sizeof(float));
    if (!im) {
        fprintf(stderr, "Failed to malloc im.\n");
        exit(EXIT_FAILURE);
    }
    int pad = 1;
    int stride = 1;
    int ksize = 3;
    if (pad != (ksize - 1) / 2) {
        printf("Pad must equal (ksize - 1) / 2.\n");
        exit(EXIT_FAILURE);
    }
    int out_w = (width + pad * 2 - ksize) / stride + 1;
    if (out_w != width) {
        printf("Output width must equal input width.\n");;
        exit(EXIT_FAILURE);
    }
    int out_h = (height + pad * 2 - ksize) / stride + 1;

    int dst_w = out_w * out_h;
    int dst_h = ksize * ksize * channels;
    int dst_n = dst_w * dst_h;
    float* col = (float*)malloc(dst_n * sizeof(float));
    if (!col) {
        fprintf(stderr, "Failed to malloc col.\n");
        exit(EXIT_FAILURE);
    }

    fill_array_rand_float(im, im_n, 0, 1);

    //pprint_mat(im, width, height, channels);

    float* d_im = 0;
    float* d_col = 0;

    CHECK_CUDA(cudaMalloc(&d_im, im_n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_col, dst_n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_im, im, sizeof(float) * im_n, cudaMemcpyHostToDevice));

    int num_cuda_kernels = (height + pad * 2) * width * channels;

    dim3 block_size(out_w, 1, 1);
    dim3 grid_size(1, out_h + 2 * pad, channels);
    size_t shared_mem_size = width * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    im2col_kernel_shared_custom KARGS(grid_size, block_size, shared_mem_size) (d_im, height, width, channels,
        ksize, d_col);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time (shared mem kernel custom): %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(col, d_col, sizeof(float) * dst_n, cudaMemcpyDeviceToHost));

    cudaFree(d_col);
    cudaFree(d_im);

    //pprint_mat(col, dst_w, dst_h, 1);
    float* col_cpu = (float*)malloc(dst_n * sizeof(float));
    if (!col_cpu) {
        fprintf(stderr, "Failed to malloc col_cpu.\n");
        exit(EXIT_FAILURE);
    }
    im2col(im, channels, height, width, ksize, pad, stride, col_cpu);

    printf("Verifiying......\n");
    for (size_t i = 0; i < dst_n; i++) {
        //printf("%f =? %f\n", col_cpu[i], col[i]);
        if (col_cpu[i] != col[i]) {
            printf("Verification Failed: i = %d, (col_cpu)%f != (col_gpu)%f\n", i, col_cpu[i], col[i]);
            wait_for_key_then_exit();
        }
    }
    printf("Verifiction Success!!!\n");
}

__global__ void im2col_shared_kernel_claude(
    const float* input, float* output,
    int channels, int height, int width,
    int kernel_h, int kernel_w, int stride, int pad,
    int out_height, int out_width) {

    // Define shared memory for a tile of the input image (including padding)
    extern __shared__ float shared_input[];

    int block_row = blockIdx.y * blockDim.y; // Start row in the output
    int block_col = blockIdx.x * blockDim.x; // Start col in the output
    int c = blockIdx.z;                      // Channel being processed

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Calculate the starting position in the input for this thread's tile
    int start_input_row = block_row * stride - pad;
    int start_input_col = block_col * stride - pad;

    // Calculate shared memory dimensions including padding
    int shared_height = blockDim.y + kernel_h - 1;
    int shared_width = blockDim.x + kernel_w - 1;

    // Load data from global memory to shared memory
    // Each thread may need to load multiple elements to cover the padded region
    for (int i = thread_row; i < shared_height; i += blockDim.y) {
        for (int j = thread_col; j < shared_width; j += blockDim.x) {
            int input_row = start_input_row + i;
            int input_col = start_input_col + j;

            if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
                shared_input[i * shared_width + j] =
                    input[c * height * width + input_row * width + input_col];
            }
            else {
                shared_input[i * shared_width + j] = 0.0f; // Zero padding
            }
        }
    }

    __syncthreads(); // Ensure all threads have loaded their data

    // Only threads within the output dimensions write results
    if (thread_row < out_height && thread_col < out_width) {
        // Calculate the index in the output matrix
        int output_index = (block_row + thread_row) * out_width + (block_col + thread_col);

        // Loop through the kernel window
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                // Calculate the position in shared memory to read from
                int local_row = thread_row + i;
                int local_col = thread_col + j;

                // Write to the output matrix
                int output_channel_offset = (c * kernel_h * kernel_w + i * kernel_w + j) * out_height * out_width;
                output[output_channel_offset + output_index] =
                    shared_input[local_row * shared_width + local_col];
            }
        }
    }
}


void cuda_test_im2col_shared_claude(void) {
    int width = 320;
    int height = 320;
    int channels = 512;
    if (width % 32 != 0) {
        printf("Input width must be a multiple of 32.\n");
        exit(EXIT_FAILURE);
    }
    int im_n = width * height * channels;
    float* im = (float*)calloc(im_n, sizeof(float));
    if (!im) {
        fprintf(stderr, "Failed to calloc im.\n");
        exit(EXIT_FAILURE);
    }
    int pad = 1;
    int stride = 1;
    int ksize = 3;
    if (pad != (ksize - 1) / 2) {
        printf("Pad must equal (ksize - 1) / 2.\n");
        exit(EXIT_FAILURE);
    }
    int out_w = (width + pad * 2 - ksize) / stride + 1;
    int out_h = (height + pad * 2 - ksize) / stride + 1;

    int dst_w = out_w * out_h;
    int dst_h = ksize * ksize * channels;
    int dst_n = dst_w * dst_h;
    float* col = (float*)calloc(dst_n, sizeof(float));
    if (!col) {
        fprintf(stderr, "Failed to calloc col.\n");
        exit(EXIT_FAILURE);
    }

    fill_array_rand_float(im, im_n, 0, 1);

    float* d_im = 0;
    float* d_col = 0;

    CHECK_CUDA(cudaMalloc(&d_im, im_n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_col, dst_n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_im, im, sizeof(float) * im_n, cudaMemcpyHostToDevice));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((out_w + blockSize.x - 1) / blockSize.x,
        (out_h + blockSize.y - 1) / blockSize.y, channels);
    size_t shared_memory_size = (blockSize.x + 2 * pad) * (blockSize.y + 2 * pad) * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    im2col_shared_kernel_claude KARGS(gridSize, blockSize, shared_memory_size) (d_im, d_col, channels, height, width,
        ksize, ksize, stride, pad, out_h, out_w);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time (shared mem kernel claude): %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(col, d_col, sizeof(float) * dst_n, cudaMemcpyDeviceToHost));
    cudaFree(d_col);
    cudaFree(d_im);
    
    float* col_cpu = (float*)calloc(dst_n, sizeof(float));
    if (!col_cpu) {
        fprintf(stderr, "Failed to calloc col_cpu.\n");
        exit(EXIT_FAILURE);
    }
    im2col(im, channels, height, width, ksize, pad, stride, col_cpu);  // gives known correct result

    printf("Verifiying......\n");
    float epsilon = 1e-5f;
    for (size_t i = 0; i < dst_n; i++) {
        //printf("%f =? %f\n", col_cpu[i], col[i]);
        if (fabs(col_cpu[i] - col[i]) > epsilon) {
            printf("Verification Failed: i = %zu, (col_cpu)%f != (col_gpu)%f\n", i, col_cpu[i], col[i]);
            wait_for_key_then_exit();
        }
    }
    printf("Verifiction Success!!!\n");
}