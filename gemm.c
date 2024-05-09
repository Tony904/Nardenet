#include "gemm.h"
#include <stdio.h>
#include <omp.h>


void gemm_v1(int M, int N, int K, float* A, float* B, float* C) {
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
				//printf("*C = %f, a = %f, *B = %f\n", *(C-1), a, *(B-1));
			}
			C = C_start;
		}
		B = B_start;
	}
	printf("\nmmm done.\n");
}

void gemm(int M, int N, int K, float* A, float* B, float* C) {
	// M = # of filters
	// N = # of convolutions/dot products performed per filter
	// K = # of elements per filter
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