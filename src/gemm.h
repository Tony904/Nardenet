#ifndef GEMM_H
#define GEMM_H

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

	/*M = # of filters
	N = # of patches (# of dot products performed per filter)
	K = # of weights per filter
	A = filter matrix (M * K)
	B = expanded input matrix (K * N)
	C = output dot products (M * N)*/
	void gemm(size_t M, size_t N, size_t K, float* A, float* B, float* C);
	void gemm_groups(size_t M, size_t N, size_t K, float* A, float* B, float* C, size_t n_groups);
	/*A[M*K], B[N*K], BT[K*N], C[M*N]*/
	void gemm_atb(size_t M, size_t N, size_t K, float* A, float* B, float* C);
	void gemm_atb_groups(size_t M, size_t N, size_t K, float* A, float* B, float* C, size_t n_groups);
	/*A[M*N], AT[N*M], B[M*K], C[N*K]*/
	void gemm_tab(size_t M, size_t N, size_t K, float* A, float* B, float* C);
	void gemm_tab_groups(size_t M, size_t N, size_t K, float* A, float* B, float* C, size_t n_groups);
	/*M = # of filters, N = out_w * out_h*/
	void add_biases(float* output, float* biases, size_t M, size_t N, size_t batch_size);
	/*M = # of filters, K = out_w * out_h*/
	void get_bias_grads(float* bias_grads, float* grads, size_t M, size_t K, size_t batch_size);

	void test_gemm_groups(void);

#ifdef __cplusplus
}
#endif
#endif
