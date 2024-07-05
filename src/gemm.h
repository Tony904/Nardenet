#ifndef GEMM_H
#define GEMM_H


#ifdef __cplusplus
extern "C" {
#endif

	/*M = # of filters
	N = # of patches (# of dot products performed per filter)
	K = # of weights per filter
	A = filter matrix (M * K)
	B = expanded input matrix (K * N)
	C = output dot products (M * N)*/
	void gemm(int M, int N, int K, float* A, float* B, float* C);
	/*A[M*K], B[N*K], BT[K*N], C[M*N]*/
	void gemm_atb(int M, int N, int K, float* A, float* B, float* C);
	/*A[M*N], AT[N*M], B[M*K], C[N*K]*/
	void gemm_tab(int M, int N, int K, float* A, float* B, float* C);
	/*M = # of filters, N = out_w * out_h*/
	void add_biases(float* output, float* biases, int M, int N);
	/*M = # of filters, K = out_w * out_h*/
	void get_bias_grads(float* bias_grads, float* grads, int M, int K);

	void test_gemm_atb(void);
	void test_gemm_tab(void);

#ifdef __cplusplus
}
#endif
#endif
