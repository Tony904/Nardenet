#ifndef XCUDA_H
#define XCUDA_H


#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "locmacro.h"
#include "image.h"


#ifndef __TIME__
#define __TIME__
#endif



#ifdef __cplusplus
extern "C" {
#endif


	void ___check_cuda(cudaError_t x, const char* const filename, const char* const funcname, const int line, const char* time);

#define CHECK_CUDA(x) ___check_cuda(x, NARDENET_LOCATION, " - " __TIME__)
#define BLOCKSIZE 512
#define GET_GRIDSIZE(n, blocksize) (n + blocksize - 1) / blocksize


	void print_gpu_props(void);

	// blas.cu
	void add_biases_gpu(float* arr, int spatial, float* biases, int n_filters, int batch_size);
	void get_bias_grads_gpu(float* bias_grads, float* grads, int n_filters, int spatial, int batch_size);
	void add_arrays_gpu(float* A, float* B, int n);
	void copy_array_gpu(float* src, float* dst, int n);
	void scale_array_gpu(float* A, float scalar, int n);
	void clamp_array_gpu(float* A, float upper, float lower, int n);
	void zero_array_gpu(float* A, int n);

	// image.cu
	void transform_colorspace_gpu(image* img, float brightness_scalar, float contrast_scalar, float saturation_scalar, float hue_shift);

	// avgpool.cu
	void launch_forward_avgpool_kernel(float* input, float* output, int spatial, int c, int batch_size);
	void launch_backward_avgpool_kernel(float* grads_x, float* grads_y, int spatial, int c, int batch_size);

	// maxpool.cu
	void launch_forward_maxpool_kernel(float* src, float* dst, int* max_indexes, int src_w, int src_h, int dst_w, int dst_h, int dst_n, int batch_size);

	// derivatives.cu
	void get_grads_sigmoid_gpu(float* grads, float* act_output, size_t out_n, size_t batch_size);
	void get_grads_mish_gpu(float* grads, float* act_inputs, size_t out_n, size_t batch_size);
	void get_grads_relu_gpu(float* grads, float* act_inputs, size_t out_n, size_t batch_size);
	void get_grads_leaky_relu_gpu(float* grads, float* act_inputs, size_t out_n, size_t batch_size);
	void get_grads_tanh_gpu(float* grads, float* act_inputs, size_t out_n, size_t batch_size);
	void regularize_l1_gpu(float* weight_grads, float* weights, size_t size, float decay);
	void regularize_l2_gpu(float* weight_grads, float* weights, size_t size, float decay);

	// im2col.cu
	void im2col_gpu(float* data_im, float* data_col, int channels, int h, int w, int ksize, int stride, int pad, int out_h, int out_w);
	void col2im_gpu(float* data_col, float* data_im, int channels, int h, int w, int ksize, int stride, int pad, int n);

	// gemm.cu
	void gemm_gpu(size_t M, size_t N, size_t K, float* A, float* B, float* C, int n_groups);
	void gemm_atb_gpu(size_t M, size_t N, size_t K, float* A, float* B, float* C, int n_groups);
	void gemm_tab_gpu(size_t M, size_t N, size_t K, float* A, float* B, float* C, int n_groups);
	void add_biases_gpu(float* arr, int spatial, float* biases, int channels, int batch_size);

	// batchnorm.cu
	void forward_batchnorm_gpu(float* gammas, float* betas, float* means, float* variances, float* rolling_means, float* rolling_variances,
		float* Z, float* Z_norm, float* act_inputs, int spatial, int n_filters, int batch_size);
	void backward_batchnorm_gpu(float* grads, float* Z, float* Z_norm, float* means, float* variances,
		float* gammas, float* gamma_grads, int spatial, int n_filters, int batch_size);

	// derivatives.cu
	void get_grads_sigmoid_gpu(float* grads, float* act_output, size_t out_n, size_t batch_size);
	void get_grads_mish_gpu(float* grads, float* act_inputs, size_t out_n, size_t batch_size);
	void get_grads_relu_gpu(float* grads, float* act_inputs, size_t out_n, size_t batch_size);
	void get_grads_leaky_relu_gpu(float* grads, float* act_inputs, size_t out_n, size_t batch_size);
	void get_grads_tanh_gpu(float* grads, float* act_inputs, size_t out_n, size_t batch_size);
	void regularize_l1_gpu(float* weight_grads, float* weights, size_t size, float decay);
	void regularize_l2_gpu(float* weight_grads, float* weights, size_t size, float decay);


#ifdef __cplusplus
}
#endif
#endif