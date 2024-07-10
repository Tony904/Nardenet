#include "gemm.h"
#include <stdio.h>
#include <omp.h>
#include "xallocs.h"


void print_test_matrix(int rows, int cols, int channels, float* matrix);


void gemm(int M, int N, int K, float* A, float* B, float* C) {
	/*M = # of filters
	N = # of patches (# of dot products performed per filter)
	K = # of weights per filter
	A = filter matrix (M * K)
	B = expanded input matrix (K * N)
	C = output dot products (M * N)*/
	int m;
#pragma omp parallel for
	for (m = 0; m < M; m++) {
		for (int k = 0; k < K; k++) {
			float a = A[m * K + k];
			for (int n = 0; n < N; n++) {
				C[m * N + n] += a * B[k * N + n];
			}
		}
	}
}

/*A[M*K], B[N*K], BT[K*N], C[M*N]*/
void gemm_atb(int M, int N, int K, float* A, float* B, float* C) {
	// M = # of filters
	// N = # of weights per filter
	// K = # of patches
	// A = M * K
	// B = N * K -> transpose -> K * N
	// C = M * N
	//printf("gemm_atb...");
	int m;
#pragma omp parallel for
	for (m = 0; m < M; m++) {
		for (int k = 0; k < K; k++) {
			float a = A[m * K + k];
			int mN = m * N;
			for (int n = 0; n < N; n++) {
				C[mN + n] += a * B[n * K + k];
			}
		}
	}
	//printf("done.\n");
}

/*A[M*N], AT[N*M], B[M*K], C[N*K]*/
void gemm_tab(int M, int N, int K, float* A, float* B, float* C) {
	// M = # of filters
	// N = # of weights per filter
	// K = # of patches
	// A = M * N -> transpose -> N * M
	// B = M * K
	// C = N * K
	int m;
#pragma omp parallel for
	for (m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			float a = A[m * N + n];
			int nK = n * K;
			int mK = m * K;
			for (int k = 0; k < K; k++) {
				C[nK + k] += a * B[mK + k];
			}
		}
	}
}

/*M = # of filters, N = out_w * out_h*/
#pragma warning(suppress:4100) // temporary
void add_biases(float* output, float* biases, int M, int N, int batch_size) {
	M = M * batch_size;
	int m;
#pragma omp parallel for
	for (m = 0; m < M; m++) {
		int mN = m * N;
		for (int n = 0; n < N; n++) {
			output[mN + n] += biases[m];
		}
	}
}

/*M = # of filters, K = out_w * out_h*/
void get_bias_grads(float* bias_grads, float* grads, int M, int K, int batch_size) {
	M = M * batch_size;
	int m;
#pragma omp parallel for
	for (m = 0; m < M; m++) {
		float sum = 0;
		int mK = m * K;
		for (int k = 0; k < K; k++) {
			sum += grads[mK + k];
		}
		bias_grads[m] += sum;  // += because they will be divided by batch count during update step
	}
}

void gemm_test(int M, int N, int K, float* A, float* B, float* C) {
	// M = # of filters
	// N = # of convolutions/dot products performed per filter
	// K = # of elements per filter
	// A = filter matrix (M * K)
	// B = expanded input matrix (K * N)
	// C = output dot products (M * N)
	float* B_start = B;
	float* C_start = C;
	for (int m = 0; m < M; m++) {
		for (int k = 0; k < K; k++) {
			float a = *(A++);
			for (int n = 0; n < N; n++) {
				*(C++) = *C + a * *(B++);
			}
			C = C_start;
		}
		B = B_start;
	}
	printf("\nmmm done.\n");
}

void test_gemm_atb(void) {
	int M = 3;
	int N = 4;
	int K = 9;
	float* A = (float*)xcalloc((size_t)(M * K), sizeof(float));
	float* B = (float*)xcalloc((size_t)(N * K), sizeof(float));
	float* C = (float*)xcalloc((size_t)(M * N), sizeof(float));
	for (int i = 0; i < M * K; i++) {
		A[i] = (float)(i + 1);
	}
	print_test_matrix(M, K, 1, A);
	for (int i = 0; i < N * K; i++) {
		B[i] = (float)(i + 2);
	}
	print_test_matrix(N, K, 1, B);
	gemm_atb(M, N, K, A, B, C);
	print_test_matrix(M, N, 1, C);
}

void test_gemm_tab(void) {
	int M = 2;
	int N = 3;
	int K = 4;
	float* A = (float*)xcalloc((size_t)(M * N), sizeof(float));
	float* B = (float*)xcalloc((size_t)(M * K), sizeof(float));
	float* C = (float*)xcalloc((size_t)(N * K), sizeof(float));
	for (int i = 0; i < M * N; i++) {
		A[i] = (float)(i + 1);
	}
	print_test_matrix(M, N, 1, A);
	for (int i = 0; i < M * K; i++) {
		B[i] = (float)(i + 2);
	}
	print_test_matrix(M, K, 1, B);
	gemm_tab(M, N, K, A, B, C);
	print_test_matrix(N, K, 1, C);
}

void print_test_matrix(int rows, int cols, int channels, float* matrix) {
	for (int ch = 0; ch < channels; ch++) {
		printf("Channel: %d\n", ch);
		for (int r = 0; r < rows; r++) {
			printf("%0.1f", matrix[ch * cols * rows + r * cols]);
			for (int c = 1; c < cols; c++) {
				printf(", %0.1f", matrix[ch * cols * rows + r * cols + c]);
			}
			printf("\n");
		}
		printf("\n");
	}
}