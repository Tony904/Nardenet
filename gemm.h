#ifndef GEMM_H
#define GEMM_H


#ifdef __cplusplus
extern "C" {
#endif

	void gemm_v1(int M, int N, int K, float* A, float* B, float* C);
	void gemm(int M, int N, int K, float* A, float* B, float* C);


#ifdef __cplusplus
}
#endif
#endif
