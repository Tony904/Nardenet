#include <stdio.h>
#include "xcuda.h"
#include "xallocs.h"
#include "utils.h"
#include "im2col.h"

#ifdef __INTELLISENSE__
#define KARGS(...)
#define cuda_syncthreads()
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#define cuda_syncthreads() __syncthreads()
#endif


__global__ void im2col_kernel(const float* data_im, const int height, const int width, const int channels,
    const int ksize, const int pad, const int stride,
    const int height_out, const int width_out, float* data_col,
    int n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
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

// Pad has to equal (ksize - 1) / 2. Stride must = 1 (for now). Block size must = input width.
__global__ void im2col_kernel_shared(const float* data_im, const int height, const int width, const int channels,
    const int ksize, const int pad, const int stride,
    const int height_col, const int width_col, float* data_col,
    int n) {

    extern __shared__ float shared_input[];  // Equals width_col * sizeof(float)

    int tx = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tx;

    // assert n == height_col * width_col * channels * ksize;

    // Each thread is responsible for copying one row of the kernel window
    if (index < n) {

        const int col_out = index % width_col;
        const int row_out = (index / width_col) % height_col;
        const int krow = (index / width_col / height_col) % ksize;
        const int ch_in = (index / width_col / height_col / ksize) % channels;

        const int row_in = row_out * stride - pad + krow;

        const int base_data_col_index = (((ch_in * ksize + krow) * ksize) * height_col + row_out) * width_col + col_out;

        shared_input[tx] = data_im[(ch_in * height + row_in) * width + tx];

        cuda_syncthreads();

        for (int kcol = 0; kcol < ksize; ++kcol) {
            int col_in = col_out * stride - pad + kcol;
            int data_col_index = base_data_col_index + kcol * height_col * width_col;

            if (row_in >= 0 && row_in < height && col_in >= 0 && col_in < width) {
                data_col[data_col_index] = shared_input[col_in];
            }
            else {
                data_col[data_col_index] = 0.0F;
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
        wait_for_key_then_exit();
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
        fprintf(stderr, "Failed to calloc col.");
        exit(EXIT_FAILURE);
    }

    fill_array_rand_float(im, im_n, 0, 1);

    float* d_im = 0;
    float* d_col = 0;

    CHECK_CUDA(cudaMalloc(&d_im, im_n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_col, dst_n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_im, im, sizeof(float) * im_n, cudaMemcpyHostToDevice));

    int threads_per_block = 512;  // Required for the shared mem im2col kernel to work
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
        fprintf(stderr, "Failed to calloc col_cpu.");
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

void cuda_test_im2col_shared(void) {
    int width = 320;
    int height = 320;
    int channels = 80;
    if (width % 32 != 0) {
        printf("Input width must be a multiple of 32.\n");
        wait_for_key_then_exit();
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
    if (width != out_w) {
        printf("Input width must be equal to output width. (%d != %d)", width, out_w);
        return;
    }
    int dst_w = out_w * out_h;
    int dst_h = ksize * ksize * channels;
    int dst_n = dst_w * dst_h;
    float* col = (float*)calloc(dst_n, sizeof(float));
    if (!col) {
        fprintf(stderr, "Failed to calloc col.");
        exit(EXIT_FAILURE);
    }

    fill_array_rand_float(im, im_n, 0, 1);

    float* d_im = 0;
    float* d_col = 0;

    CHECK_CUDA(cudaMalloc(&d_im, im_n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_col, dst_n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_im, im, sizeof(float) * im_n, cudaMemcpyHostToDevice));

    int threads_per_block = out_w;  // Required for the shared mem im2col kernel to work
    int num_cuda_kernels = out_h * out_w * channels * ksize;
    int num_blocks = (num_cuda_kernels + threads_per_block - 1) / threads_per_block;

    size_t shared_mem_size = out_w * sizeof(float);
    printf("num_blocks = %d, width = %d, shared_mem_size = %zu\n", num_blocks, width, shared_mem_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    im2col_kernel_shared KARGS(num_blocks, threads_per_block, shared_mem_size) (d_im, height, width, channels,
        ksize, pad, stride,
        out_h, out_w, d_col,
        num_cuda_kernels);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time (shared mem kernel): %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(col, d_col, sizeof(float) * dst_n, cudaMemcpyDeviceToHost));
    
    //pprint_mat(col, dst_w, dst_h, 1);
    float* col_cpu = (float*)calloc(dst_n, sizeof(float));
    if (!col_cpu) {
        fprintf(stderr, "Failed to calloc col_cpu.");
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