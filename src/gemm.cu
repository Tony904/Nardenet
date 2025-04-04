#include <stdio.h>
#include "xcuda.h"
#include "xallocs.h"
#include "utils.h"
#include "gemm.h"
#include <math.h>


#ifdef __INTELLISENSE__
#define KARGS(...)
#define __syncthreads()
#else
#define KARGS(...) <<< __VA_ARGS__ >>>
#endif


void print_test_matrix(size_t rows, size_t cols, size_t channels, float* matrix);


#define TILE_SIZE 16

__global__ void gemm_shared(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int M, int N, int K,
	float alpha,
	float beta)
{
	// Shared memory for A and B tiles
	__shared__ float Asub[TILE_SIZE][TILE_SIZE];
	__shared__ float Bsub[TILE_SIZE][TILE_SIZE];

	// Row and column indices of the C element to work on
	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;

	float cValue = 0.0f;

	// Loop over tiles of A and B to accumulate the result
	for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
		// Load tiles into shared memory
		int Arow = row;
		int Acol = t * TILE_SIZE + threadIdx.x;

		int Brow = t * TILE_SIZE + threadIdx.y;
		int Bcol = col;

		Asub[threadIdx.y][threadIdx.x] = (Arow < M && Acol < K) ? A[Arow * K + Acol] : 0.0f;
		Bsub[threadIdx.y][threadIdx.x] = (Brow < K && Bcol < N) ? B[Brow * N + Bcol] : 0.0f;

		__syncthreads();

		// Multiply the two tiles
		for (int i = 0; i < TILE_SIZE; ++i) {
			cValue += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];
		}

		__syncthreads();
	}

	// Write the result back to global memory
	if (row < M && col < N) {
		C[row * N + col] = alpha * cValue + beta * C[row * N + col];
	}
}



void cuda_test_gemm(void) {
	/*
	M = # of filters
	N = # of outputs per filter
	K = # of weights per filter (if n_groups = 1)
	A = weight matrix (M * K)
	B = expanded input matrix (K * N)
	C = output dot products (M * N)
	*/
	int width = 320;
	int height = width;
	int channels = 80;
	if (width % 32 != 0) {
		printf("Input width must be a multiple of 32.\n");
		exit(EXIT_FAILURE);
	}

	size_t n_groups = 1;
	size_t n_filters = 128;
	size_t pad = 1;
	size_t stride = 1;
	size_t ksize = 3;
	size_t out_size = (width + 2 * pad - ksize) / stride + 1; // square image
	size_t out_n = ksize * ksize * channels * out_size * out_size;

	size_t M = n_filters;
	size_t N = out_size * out_size;
	size_t K = ksize * ksize * channels;
	
	float* A = (float*)calloc((size_t)(M * K / n_groups), sizeof(float));
	float* B = (float*)calloc((size_t)(N * K), sizeof(float));
	float* C = (float*)calloc((size_t)(M * N), sizeof(float));
	for (size_t i = 0; i < M * K; i++) {
		A[i] = (float)(i + 1);
	}
	//print_test_matrix(M, K / n_groups, 1, A);
	for (size_t i = 0; i < N * K; i++) {
		B[i] = (float)(i + 2);
	}

	float* d_a = 0;
	float* d_b = 0;
	float* d_c = 0;

	CHECK_CUDA(cudaMalloc(&d_a, M * K / n_groups * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_b, N * K * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(float)));

	CHECK_CUDA(cudaMemcpy(d_a, A, M * K / n_groups * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_b, B, N * K * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemset(d_c, 0, M * N * sizeof(float)));

	dim3 threads(TILE_SIZE, TILE_SIZE);
	dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
		(M + TILE_SIZE - 1) / TILE_SIZE);

	float alpha = 1.0F;
	float beta = 0.0F;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	gemm_shared <<< blocks, threads >> > (d_a, d_b, d_c, M, N, K, alpha, beta);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("gemm kernel execution time: %f ms\n", milliseconds);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	CHECK_CUDA(cudaGetLastError());

	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaMemcpy(C, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(d_a));
	CHECK_CUDA(cudaFree(d_b));
	CHECK_CUDA(cudaFree(d_c));
	

	float* gemm_cpu = (float*)calloc(M * N, sizeof(float));
	if (!gemm_cpu) {
		fprintf(stderr, "Failed to calloc im_cpu.\n");
		exit(EXIT_FAILURE);
	}
	gemm_groups(M, N, K, A, B, gemm_cpu, n_groups);
	free(A);
	free(B);
	float epsilon = 1e-5f;
	printf("Verifiying......\n");
	for (size_t i = 0; i < M * N; i++) {
		if (fabs(gemm_cpu[i] - C[i]) > epsilon) {
			printf("Verification Failed: i = %zu, (gemm_cpu)%f != (gemm_gpu)%f\n", i, gemm_cpu[i], C[i]);
			wait_for_key_then_exit();
		}
	}
	printf("Verifiction Success!!!\n");
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