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
__global__ void col2im_kernel(const float* __restrict__ data_col,
    const int width_col, const int height_col,
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

    size_t pad = 1;
    size_t stride = 1;
    size_t ksize = 3;
    size_t col_size = (width + 2 * pad - ksize) / stride + 1; // square image
    size_t col_n = ksize * ksize * channels * col_size * col_size;
    float* col = (float*)malloc((size_t)col_n * sizeof(float));
    if (!col) {
        fprintf(stderr, "Failed to malloc col.\n");
        exit(EXIT_FAILURE);
    }
    fill_array_rand_float(col, col_n, 0., 1.);
    
    int im_n = width * height * channels;
    float* im = (float*)malloc(im_n * sizeof(float));
    if (!im) {
        fprintf(stderr, "Failed to malloc im.\n");
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
    CHECK_CUDA(cudaFree(d_col));
    CHECK_CUDA(cudaFree(d_im));

    //pprint_mat(col, dst_w, dst_h, 1);
    float* im_cpu = (float*)calloc(im_n, sizeof(float));
    if (!im_cpu) {
        fprintf(stderr, "Failed to calloc im_cpu.\n");
        exit(EXIT_FAILURE);
    }
    col2im(col, channels, height, width, ksize, pad, stride, im_cpu);

    float epsilon = 1e-5f;
    printf("Verifiying......\n");
    for (size_t i = 0; i < im_n; i++) {
        //printf("%f =? %f\n", col_cpu[i], col[i]);
        if (fabs(im_cpu[i] - im[i]) > epsilon) {
            printf("Verification Failed: i = %zu, (im_cpu)%f != (im_gpu)%f\n", i, im_cpu[i], im[i]);
            wait_for_key_then_exit();
        }
    }
    printf("Verifiction Success!!!\n");
}

/**
 * Wrapper function to launch the col2im CUDA kernel with appropriate grid and block dimensions
 */
//void launch_col2im_kernel(const float* col_data, float* im_data,
//    int batch_size, int channels, int height, int width,
//    int kernel_size, int pad, int stride) {
//    // Calculate output dimensions
//    int output_h = (height + 2 * pad - kernel_size) / stride + 1;
//    int output_w = (width + 2 * pad - kernel_size) / stride + 1;
//
//    // Define block and grid dimensions
//    const int TILE_SIZE = 16;
//    dim3 block(TILE_SIZE, TILE_SIZE);
//    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE,
//        (height + TILE_SIZE - 1) / TILE_SIZE,
//        channels);
//
//    // Launch kernel for each batch
//    for (int n = 0; n < batch_size; ++n) {
//        float* batch_im_data = im_data + n * channels * height * width;
//        col2im_kernel_tiled << <grid, block >> > (col_data, batch_im_data, height, width,
//            channels, kernel_size, pad, stride,
//            output_h, output_w);
//    }
//}

/*******************************************
                   IM2COL
*******************************************/

__global__ void im2col_kernel_no_share(const float* __restrict__ data_im,
    const int height, const int width, const int channels,
    const int ksize, const int pad, const int stride,
    const int height_out, const int width_out,
    float* __restrict__ data_col,
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

void cuda_test_im2col_no_share(void) {
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

    im2col_kernel_no_share KARGS(num_blocks, threads_per_block) (d_im, height, width, channels,
        ksize, pad, stride,
        out_h, out_w, d_col,
        num_cuda_kernels);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time (no shared): %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(col, d_col, sizeof(float) * dst_n, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_col));
    CHECK_CUDA(cudaFree(d_im));

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

__global__ void im2col_kernel_shared(
    const float* __restrict__ input, float* __restrict__ output,
    const int channels, const int height, const int width,
    const int ksize, const int stride, const int pad,
    const int out_height, const int out_width) {

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
    int shared_height = blockDim.y + ksize - 1;
    int shared_width = blockDim.x + ksize - 1;

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
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                // Calculate the position in shared memory to read from
                int local_row = thread_row + i;
                int local_col = thread_col + j;

                // Write to the output matrix
                int output_channel_offset = (c * ksize * ksize + i * ksize + j) * out_height * out_width;
                output[output_channel_offset + output_index] =
                    shared_input[local_row * shared_width + local_col];
            }
        }
    }
}


void cuda_test_im2col_shared(void) {
    int width = 512;
    int height = 512;
    int channels = 128;
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

    im2col_kernel_shared KARGS(gridSize, blockSize, shared_memory_size) (d_im, d_col, channels, height, width,
        ksize, stride, pad, out_h, out_w);

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
    CHECK_CUDA(cudaFree(d_col));
    CHECK_CUDA(cudaFree(d_im));
    
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