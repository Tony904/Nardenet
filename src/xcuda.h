#ifndef XCUDA_H
#define XCUDA_H

//#define GPU

#include <stdlib.h>
#ifndef __TIME__
#define __TIME__
#endif

#ifdef GPU
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include "locmacro.h"
#include "image.h"
#endif


#ifdef __cplusplus
extern "C" {
#endif

	void gpu_not_defined(void);
	void activate_cuda_alloc_tracking(void);
	void print_cuda_alloc_list(void);

#ifndef GPU
#define CHECK_CUDA(x) gpu_not_defined()
#define CUDA_MALLOC(pPtr, size) gpu_not_defined()
#define CUDA_FREE(p) gpu_not_defined()
#define CUDA_MEMCPY_H2D(dst, src, size) gpu_not_defined()
#define CUDA_MEMCPY_D2H(dst, src, size) gpu_not_defined()
#else
	void ___check_cuda(cudaError_t x, const char* const filename, const char* const funcname, const int line, const char* time);
	void ___check_cublas(cublasStatus_t x, const char* const filename, const char* const funcname, const int line, const char* time);
	void ___cudaMalloc(void** devPtr, size_t num_elements, size_t size_per_element, const char* const filename, const char* const funcname, const int line, const char* time);
	void ___cudaFree(void** devPtr, const char* const filename, const char* const funcname, const int line, const char* time);
	void ___cudaMemcpy(void* dst, void* src, size_t size, enum cudaMemcpyKind kind, const char* const filename, const char* const funcname, const int line, const char* time);
#define CHECK_CUDA(x) ___check_cuda(x, NARDENET_LOCATION, " - " __TIME__)
#define CHECK_CUBLAS(x) ___check_cublas(x, NARDENET_LOCATION, " - " __TIME__)
#define CUDA_MALLOC(pPtr, n, s) ___cudaMalloc(pPtr, n, s, NARDENET_LOCATION, " - " __TIME__)
#define CUDA_FREE(pPtr) ___cudaFree((void**)p, NARDENET_LOCATION, " - " __TIME__)
#define CUDA_MEMCPY_H2D(dst, src, size) ___cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice, NARDENET_LOCATION, " - " __TIME__)
#define CUDA_MEMCPY_D2H(dst, src, size) ___cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost, NARDENET_LOCATION, " - " __TIME__)
#define BLOCKSIZE 512
#define GET_GRIDSIZE(n, blocksize) (n + blocksize - 1) / blocksize

	cublasHandle_t get_cublas_handle(void);
	void print_gpu_props(void);
	void print_gpu_float_array(float* gpu_array, size_t size, char* text);
	void compare_cpu_gpu_arrays(float* cpu_array, float* gpu_array, size_t size, int layer_id, char* text);
	void print_cpu_gpu_arrays(float* cpu_array, float* gpu_array, size_t size, char* text);

	// blas.cu
	void add_biases_gpu(float* output, float* biases, int n_filters, int spatial, int batch_size);
	void get_bias_grads_gpu(float* bias_grads, float* grads, int n_filters, int spatial, int batch_size);
	void dot_product_gpu(float* A, float* B, int n, float* result);
	void sum_array_gpu(float* A, int n, float* sum);
	void add_arrays_gpu(float* X, float* Y, int n);
	void copy_array_gpu(float* src, float* dst, int n);
	void scale_add_array_gpu(float* src, float* dst, float* scalar, int n);
	void clamp_array_gpu(float* A, float upper, float lower, int n);
	void zero_array_gpu(float* A, int n);

	// image.cu
	void transform_colorspace_gpu(image* img, float brightness_scalar, float contrast_scalar, float saturation_scalar, float hue_shift);

	// avgpool.cu
	void launch_forward_avgpool_kernel(float* input, float* output, int spatial, int c, int batch_size);
	void launch_backward_avgpool_kernel(float* grads_x, float* grads_y, int spatial, int c, int batch_size);

	// maxpool.cu
	void launch_forward_maxpool_general_kernel(float* input, float* output, float* grads, float** max_ptrs, int src_w, int src_h, int dst_w, int dst_h, int dst_n, int ksize, int stride);
	void launch_forward_maxpool_standard_kernel(float* input, float* output, float* grads, float** max_ptrs, int src_w, int src_h, int dst_w, int dst_h, int dst_n);
	void launch_backward_maxpool_kernel(float* grads, float** max_ptrs, int n);

	// upsample.cu
	void launch_forward_upsample_kernel(float* input, float* output, int w, int h, int c, int ksize, int batch_size);
	void launch_backward_upsample_kernel(float* grads_x, float* grads_y, int w, int h, int c, int ksize, int batch_size);

	// derivatives.cu
	void get_grads_sigmoid_gpu(float* grads, float* act_output, int out_n, int batch_size);
	void get_grads_mish_gpu(float* grads, float* act_inputs, int out_n, int batch_size);
	void get_grads_relu_gpu(float* grads, float* act_inputs, int out_n, int batch_size);
	void get_grads_leaky_relu_gpu(float* grads, float* act_inputs, int out_n, int batch_size);
	void get_grads_tanh_gpu(float* grads, float* act_inputs, int out_n, int batch_size);
	void regularize_l1_gpu(float* weight_grads, float* weights, size_t size, float decay);
	void regularize_l2_gpu(float* weight_grads, float* weights, size_t size, float decay);

	// im2col.cu
	void im2col_gpu(float* data_im, float* data_col, int w, int h, int c, int out_w, int out_h, int ksize, int stride, int pad);
	void col2im_gpu(float* data_col, float* data_im, int w, int h, int out_w, int out_h, int ksize, int stride, int pad, int n);

	// gemm.cu
	void gemm_gpu(int M, int N, int K, float* A, float* B, float* C, int n_groups);
	void gemm_atb_gpu(int M, int N, int K, float* A, float* B, float* C, int n_groups);
	void gemm_tab_gpu(int M, int N, int K, float* A, float* B, float* C, int n_groups);

	// batchnorm.cu
	void forward_batchnorm_gpu(float* gammas, float* betas, float* means, float* variances, float* rolling_means, float* rolling_variances, float* Z, float* Z_norm, float* act_inputs, int spatial, int n_filters, int batch_size);
	void backward_batchnorm_gpu(float* grads, float* Z, float* Z_norm, float* means, float* variances, float* gammas, float* gamma_grads, int spatial, int n_filters, int batch_size);

	// loss.cu
	void launch_loss_mae_kernel(float* grads, float* output, float* truth, float* errors, int n, int batch_size);
	void launch_loss_mse_kernel(float* grads, float* output, float* truth, float* errors, int n, int batch_size);
	void launch_loss_cce_kernel(float* grads, float* output, float* truth, float* errors, int n, int batch_size);
	void launch_loss_bce_kernel(float* grads, float* output, float* truth, float* errors, int n, int batch_size);
	void launch_loss_l1_kernel(float* weights, size_t n, float decay, float* loss);
	void launch_loss_l2_kernel(float* weights, size_t n, float decay, float* loss);

	// update.cu
	void launch_update_kernel(float* vals, float* grads, float* velocities, int n_vals, float momentum, float rate);

	// activations.cu
	void activate_mish_gpu(float* Z, float* output, size_t size, size_t batch_size);
	void activate_relu_gpu(float* Z, float* output, size_t size, size_t batch_size);
	void activate_sigmoid_gpu(float* Z, float* output, size_t size, size_t batch_size);
	void activate_leaky_relu_gpu(float* Z, float* output, size_t size, size_t batch_size);
	void activate_tanh_gpu(float* Z, float* output, size_t out_n, size_t batch_size);
	void activate_softmax_gpu(float* Z, float* output, size_t size, size_t batch_size);
	void test_activate_softmax_gpu(void);


#endif  // GPU

#ifdef __cplusplus
}
#endif
#endif