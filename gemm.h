#ifndef GEMM_H
#define GEMM_H


#ifdef __cplusplus
extern "C" {
#endif

	void gemm(int M, int N, int K, float* A, float* B, float* C);
	void gemm_atb(int M, int N, int K, float* A, float* B, float* C);
	void add_biases(float* output, float* biases, int M, int N);

	void gemm_atb_test(void);

#ifdef __cplusplus
}
#endif
#endif
