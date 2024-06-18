#include "gemm.h"
#include <stdio.h>
#include <omp.h>
#include "xallocs.h"


void print_test_matrix(int rows, int cols, int channels, float* matrix);


void gemm(int M, int N, int K, float* A, float* B, float* C) {
	// M = # of filters
	// N = # of patches (# of dot products performed per filter)
	// K = # of weights per filter
	// A = filter matrix (M * K)
	// B = expanded input matrix (K * N)
	// C = output dot products (M * N)
	printf("gemm... ");
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
	printf("done.\n");
}

void gemm_atb(int M, int N, int K, float* A, float* B, float* C) {
	// M = # of filters
	// N = # of weights per filter
	// K = # of patches
	// A = M * K
	// B = N * K -> transpose -> K * N
	// C = M * N
	printf("gemm_atb...");
	int m;
	for (m = 0; m < M; m++) {
		for (int k = 0; k < K; k++) {
			float a = A[m * K + k];
			for (int n = 0; n < N; n++) {
				C[m * N + n] += a * B[n * K + k];
			}
		}
	}
	printf("done.\n");
}

void add_biases(float* output, float* biases, int M, int N) {
	// M = # of filters (aka out_c)
	// N = out_w * out_h
	int m;
	#pragma omp parallel for
	for (m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			output[m * N + n] += biases[m];
		}
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

void gemm_atb_test(void) {
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