#include "gemm.h"
#include <stdio.h>
#include <omp.h>
#include "xallocs.h"


void print_test_matrix(size_t rows, size_t cols, size_t channels, float* matrix);


void gemm(size_t M, size_t N, size_t K, float* A, float* B, float* C) {
	/*
	M = # of filters
	N = # of patches (# of dot products performed per filter)
	K = # of weights per filter
	A = filter matrix (M * K)
	B = expanded input matrix (K * N)
	C = output dot products (M * N)
	*/
	size_t m;
#pragma omp parallel for
	for (m = 0; m < M; m++) {
		size_t mK = m * K;
		size_t mN = m * N;
		for (size_t k = 0; k < K; k++) {
			float a = A[mK + k];
			size_t kN = k * N;
			for (size_t n = 0; n < N; n++) {
				C[mN + n] += a * B[kN + n];
			}
		}
	}
}

void gemm_groups(size_t M, size_t N, size_t K, float* A, float* B, float* C, size_t n_groups) {
	/*
	M = # of filters
	N = # of outputs per filter
	K = # of weights per filter (as if n_groups = 1)
	A = weight matrix (M * K)
	B = expanded input matrix (K * N)
	C = output dot products (M * N)
	*/
	M = M / n_groups;  // # of filters per group
	K = K / n_groups;  // # of weights per filter per group
	size_t MK = M * K;
	size_t MN = M * N;
	size_t NK = N * K;
	size_t g;
#pragma omp parallel for firstprivate(MK, MN, NK)
	for (g = 0; g < n_groups; g++) {
		size_t gMK = g * MK;
		size_t gMN = g * MN;
		size_t gNK = g * NK;
		for (size_t m = 0; m < M; m++) {
			size_t gMNmN = gMN + m * N;
			size_t gMKmK = gMK + m * K;
			for (size_t k = 0; k < K; k++) {
				float a = A[gMKmK + k];
				size_t gNKkN = gNK + k * N;
				for (size_t n = 0; n < N; n++) {
					C[gMNmN + n] += a * B[gNKkN + n];
				}
			}
		}
	}
}

/*A[M*K], B[N*K], BT[K*N], C[M*N]*/
void gemm_atb(size_t M, size_t N, size_t K, float* A, float* B, float* C) {
	// M = # of filters
	// N = # of weights per filter
	// K = # of patches
	// A = M * K
	// B = N * K -> transpose -> K * N
	// C = M * N
	size_t m;
#pragma omp parallel for
	for (m = 0; m < M; m++) {
		size_t mK = m * K;
		size_t mN = m * N;
		for (size_t k = 0; k < K; k++) {
			float a = A[mK + k];
			for (size_t n = 0; n < N; n++) {
				C[mN + n] += a * B[n * K + k];
			}
		}
	}
}

/*A[M*K], B[N*K], BT[K*N], C[M*N]*/
void gemm_atb_groups(size_t M, size_t N, size_t K, float* A, float* B, float* C, size_t n_groups) {
	// M = # of filters
	// N = # of weights per filter (as if n_groups = 1)
	// K = # of outputs per filter
	// A = M * K (dC/dz grads)
	// B = N * K -> transpose -> K * N
	// C = M * N
	M = M / n_groups; // # of filters per group
	N = N / n_groups; // # of weights per filter per group
	size_t MN = M * N;
	size_t NK = N * K;
	size_t MK = M * K;
	size_t g;
#pragma omp parallel for firstprivate(MN, NK, MK)
	for (g = 0; g < n_groups; g++) {
		size_t gMN = g * MN;
		size_t gNK = g * NK;
		size_t gMK = g * MK;
		for (size_t m = 0; m < M; m++) {
			size_t gMKmK = gMK + m * K;
			size_t gMNmN = gMN + m * N;
			for (size_t k = 0; k < K; k++) {
				float a = A[gMKmK + k];
				size_t gNKk = gNK + k;
				for (size_t n = 0; n < N; n++) {
					C[gMNmN + n] += a * B[gNKk + n * K];
				}
			}
		}
	}
}

/*A[M*N], AT[N*M], B[M*K], C[N*K]*/
void gemm_tab(size_t M, size_t N, size_t K, float* A, float* B, float* C) {
	// M = # of filters
	// N = # of weights per filter
	// K = # of outputs per filter
	// A = M * N -> transpose -> N * M (weights)
	// B = M * K (dC/dz grads)
	// C = N * K (col'd array to go through col2im)
	size_t m;
#pragma omp parallel for
	for (m = 0; m < M; m++) {
		size_t mN = m * N;
		size_t mK = m * K;
		for (size_t n = 0; n < N; n++) {
			float a = A[mN + n];
			size_t nK = n * K;
			for (size_t k = 0; k < K; k++) {
				C[nK + k] += a * B[mK + k];
			}
		}
	}
}

void gemm_tab_groups(size_t M, size_t N, size_t K, float* A, float* B, float* C, size_t n_groups) {
	// M = # of filters
	// N = # of weights per filter (as if n_groups = 1)
	// K = # of outputs per filter
	// A = M * N -> transpose -> N * M (weights)
	// B = M * K (dC/dz grads)
	// C = N * K (col'd array to go through col2im)
	M = M / n_groups; // # of filters per group
	N = N / n_groups; // # of weights per filter per group
	size_t MN = M * N;
	size_t NK = N * K;
	size_t MK = M * K;
	size_t g;
#pragma omp parallel for firstprivate(MN, NK, MK)
	for (g = 0; g < n_groups; g++) {
		size_t gMN = g * MN;
		size_t gNK = g * NK;
		size_t gMK = g * MK;
		for (size_t m = 0; m < M; m++) {
			size_t gMNmN = gMN + m * N;
			size_t gMKmK = gMK + m * K;
			for (size_t n = 0; n < N; n++) {
				float a = A[gMNmN + n];
				size_t gNKnK = gNK + n * K;
				for (size_t k = 0; k < K; k++) {
					C[gNKnK + k] += a * B[gMKmK + k];
				}
			}
		}
	}
}

/*F = # of filters, S = out_w * out_h*/
#pragma warning(suppress:4100) // temporary
void add_biases(float* output, float* biases, size_t F, size_t S, size_t batch_size) {
	size_t B = (size_t)batch_size;
	size_t out_n = F * S;
	size_t f;
#pragma omp parallel for firstprivate(out_n)
	for (f = 0; f < F; f++) {
		size_t fS = f * S;
		for (size_t b = 0; b < B; b++) {
			size_t offset = b * out_n + fS;
			for (size_t s = 0; s < S; s++) {
				output[offset + s] += biases[f];
			}
		}
	}
}

/*M = # of filters, K = out_w * out_h*/
void get_bias_grads(float* bias_grads, float* grads, size_t F, size_t S, size_t batch_size) {
	size_t B = batch_size;
	size_t out_n = F * S;
	size_t f;
#pragma omp parallel for firstprivate(out_n)
	for (f = 0; f < F; f++) {
		float sum = 0.0F;
		size_t fS = f * S;
		for (size_t b = 0; b < B; b++) {
			size_t offset = b * out_n + fS;
			for (size_t s = 0; s < S; s++) {
				sum += grads[offset + s];
			}
		}
		bias_grads[f] += sum;  // += because they will be divided by batch count during update step
	}
}

void test_gemm_groups(void) {
	/*
	M = # of filters
	N = # of outputs per filter
	K = # of weights per filter (if n_groups = 1)
	A = weight matrix (M * K)
	B = expanded input matrix (K * N)
	C = output dot products (M * N)
	*/
	size_t M = 4;
	size_t N = 4;
	size_t K = 16;
	size_t n_groups = 2;
	float* A = (float*)xcalloc((size_t)(M * K / n_groups), sizeof(float));
	float* B = (float*)xcalloc((size_t)(N * K), sizeof(float));
	float* C = (float*)xcalloc((size_t)(M * N), sizeof(float));
	for (size_t i = 0; i < M * K; i++) {
		A[i] = (float)(i + 1);
	}
	print_test_matrix(M, K / n_groups, 1, A);
	for (size_t i = 0; i < N * K; i++) {
		B[i] = (float)(i + 2);
	}
	print_test_matrix(K, N, 1, B);
	gemm_groups(M, N, K, A, B, C, n_groups);
	print_test_matrix(M, N, 1, C);
}

void test_gemm_atb(void) {
	size_t M = 3;
	size_t N = 4;
	size_t K = 9;
	float* A = (float*)xcalloc(M * K, sizeof(float));
	float* B = (float*)xcalloc(N * K, sizeof(float));
	float* C = (float*)xcalloc(M * N, sizeof(float));
	for (size_t i = 0; i < M * K; i++) {
		A[i] = (float)(i + 1);
	}
	print_test_matrix(M, K, 1, A);
	for (size_t i = 0; i < N * K; i++) {
		B[i] = (float)(i + 2);
	}
	print_test_matrix(N, K, 1, B);
	gemm_atb(M, N, K, A, B, C);
	print_test_matrix(M, N, 1, C);
}

void test_gemm_atb_groups(void) {
	// A = M * K (dC/dz grads)
	// B = N * K -> transpose -> K * N
	// C = M * N
	size_t n_groups = 2;
	size_t M = 2;	// # of filters
	size_t N = 16;	// # of weights per filter (as if n_groups = 1)
	size_t K = 4;	// # of outputs per filter
	if (M % n_groups > 0 || N % n_groups > 0) {
		printf("Cannot divide filters or weights evenly between groups.\n");
		(void)getchar();
		exit(EXIT_FAILURE);
	}
	float* A = (float*)xcalloc(M * K, sizeof(float));
	float* B = (float*)xcalloc(N * K, sizeof(float));
	float* C = (float*)xcalloc(M * (N / n_groups), sizeof(float));
	for (size_t i = 0; i < M * K; i++) A[i] = (float)(i + 1);
	print_test_matrix(M, K, 1, A);
	for (size_t i = 0; i < N * K; i++) B[i] = (float)(i + 2);
	print_test_matrix(N, K, 1, B);
	gemm_atb_groups(M, N, K, A, B, C, n_groups);
	print_test_matrix(M, N / n_groups, 1, C);
}

void test_gemm_tab(void) {
	size_t M = 2;
	size_t N = 3;
	size_t K = 4;
	float* A = (float*)xcalloc(M * N, sizeof(float));
	float* B = (float*)xcalloc(M * K, sizeof(float));
	float* C = (float*)xcalloc(N * K, sizeof(float));
	for (size_t i = 0; i < M * N; i++) A[i] = (float)(i + 1);
	print_test_matrix(M, N, 1, A);
	for (size_t i = 0; i < M * K; i++) B[i] = (float)(i + 2);
	print_test_matrix(M, K, 1, B);
	gemm_tab(M, N, K, A, B, C);
	print_test_matrix(N, K, 1, C);
}

void test_gemm_tab_groups(void) {
	// A = M * N -> transpose -> N * M (weights)
	// B = M * K (dC/dz grads)
	// C = N * K (col'd array to go through col2im)
	size_t n_groups = 2;
	size_t M = 4;	// # of filters
	size_t N = 16;  // # of weights per filter (as if n_groups = 1)
	size_t K = 4;	// # of outputs per filter
	if (M % n_groups > 0 || N % n_groups > 0) {
		printf("Cannot divide filters or weights evenly between groups.\n");
		(void)getchar();
		exit(EXIT_FAILURE);
	}
	float* A = (float*)xcalloc(M * (N / n_groups), sizeof(float));
	float* B = (float*)xcalloc(M * K, sizeof(float));
	float* C = (float*)xcalloc(N * K, sizeof(float));
	for (size_t i = 0; i < M * N; i++) A[i] = (float)(i + 1);
	print_test_matrix(M, N / n_groups, 1, A);
	for (size_t i = 0; i < M * K; i++) B[i] = (float)(i + 2);
	print_test_matrix(M, K, 1, B);
	gemm_tab_groups(M, N, K, A, B, C, n_groups);
	print_test_matrix(N, K, 1, C);
}

void print_test_matrix(size_t rows, size_t cols, size_t channels, float* matrix) {
	for (size_t ch = 0; ch < channels; ch++) {
		printf("Channel: %zu\n", ch);
		for (size_t r = 0; r < rows; r++) {
			printf("%0.1f", matrix[ch * cols * rows + r * cols]);
			for (size_t c = 1; c < cols; c++) {
				printf(", %0.1f", matrix[ch * cols * rows + r * cols + c]);
			}
			printf("\n");
		}
		printf("\n");
	}
}