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


__global__ void gemm_kernel_shared(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int M, int N, int K,
	int A_offset, int B_offset, int C_offset)
{
	// Shared memory for A and B tiles
	__shared__ float A_shared[TILE_SIZE][TILE_SIZE];
	__shared__ float B_shared[TILE_SIZE][TILE_SIZE];

	// Row and column indices of the C element to work on
	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;

	float c_partial = 0.0F;

	// Loop over tiles of A and B to accumulate the result
	for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
		// Load tiles into shared memory
		int A_row = row;
		int A_col = t * TILE_SIZE + threadIdx.x;

		int B_row = t * TILE_SIZE + threadIdx.y;
		int B_col = col;

		A_shared[threadIdx.y][threadIdx.x] = (A_row < M && A_col < K) ? A[A_offset + A_row * K + A_col] : 0.0F;
		B_shared[threadIdx.y][threadIdx.x] = (B_row < K && B_col < N) ? B[B_offset + B_row * N + B_col] : 0.0F;

		__syncthreads();

		// Multiply the two tiles
		for (int i = 0; i < TILE_SIZE; ++i) {
			c_partial += A_shared[threadIdx.y][i] * B_shared[i][threadIdx.x];
		}

		__syncthreads();
	}

	// Write the result back to global memory
	if (row < M && col < N) {
		C[C_offset + row * N + col] += c_partial;
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
	int channels = 32;
	if (width % 32 != 0) {
		printf("Input width must be a multiple of 32.\n");
		exit(EXIT_FAILURE);
	}

	size_t n_groups = 2;
	size_t n_filters = 32;
	size_t pad = 1;
	size_t stride = 1;
	size_t ksize = 3;
	size_t out_size = (width + 2 * pad - ksize) / stride + 1; // square image
	size_t out_n = ksize * ksize * channels * out_size * out_size;

	size_t M = n_filters;
	size_t N = out_size * out_size;
	size_t K = ksize * ksize * channels;

	if (M % n_groups > 0 || N % n_groups > 0) {
		printf("Cannot divide filters or weights evenly between groups.\n");
		(void)getchar();
		exit(EXIT_FAILURE);
	}
	
	float* A = (float*)xcalloc((size_t)(M * K / n_groups), sizeof(float));
	float* B = (float*)xcalloc((size_t)(N * K), sizeof(float));
	float* C = (float*)xcalloc((size_t)(M * N), sizeof(float));
	fill_array_rand_float(A, M * K / n_groups, 0., 1.);
	fill_array_rand_float(B, N * K, 0., 1.);

	float* d_a = 0;
	float* d_b = 0;
	float* d_c = 0;

	CHECK_CUDA(cudaMalloc(&d_a, M * K / n_groups * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_b, N * K * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(float)));

	CHECK_CUDA(cudaMemcpy(d_a, A, M * K / n_groups * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_b, B, N * K * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemset(d_c, 0, M * N * sizeof(float)));

	M = M / n_groups;  // # of filters per group
	K = K / n_groups;  // # of weights per filter per group

	dim3 threads(TILE_SIZE, TILE_SIZE);
	dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	for (int g = 0; g < n_groups; g++) {
		int a_offset = g * M * K;
		int b_offset = g * N * K;
		int c_offset = g * M * N;
		gemm_kernel_shared KARGS(blocks, threads) (d_a, d_b, d_c, M, N, K, a_offset, b_offset, c_offset);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("gemm kernel execution time: %f ms\n", milliseconds);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());

	M = M * n_groups;
	K = K * n_groups;

	CHECK_CUDA(cudaMemcpy(C, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(d_a));
	CHECK_CUDA(cudaFree(d_b));
	CHECK_CUDA(cudaFree(d_c));

	float* gemm_cpu = (float*)xcalloc(M * N, sizeof(float));
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

/*A[M*K], B[N*K], BT[K*N], C[M*N]*/
__global__ void gemm_atb_kernel_shared(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int M, int N, int K)
{
	// Shared memory for A and B tiles
	__shared__ float A_shared[TILE_SIZE][TILE_SIZE];
	__shared__ float B_shared[TILE_SIZE][TILE_SIZE];

	// Row and column indices of the C element to work on
	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;

	float c_partial = 0.0F;

	// Loop over tiles of A and B to accumulate the result
	for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
		// Load tile of A into shared memory
		int A_row = row;
		int A_col = t * TILE_SIZE + threadIdx.x;
		A_shared[threadIdx.y][threadIdx.x] = (A_row < M && A_col < K) ? A[A_row * K + A_col] : 0.0F;

		// Load tile of B into shared memory with transposition
		// For transposition, we swap row and column when accessing global memory
		int B_col = t * TILE_SIZE + threadIdx.y; // Transposed row becomes column
		int B_row = col;                         // Transposed column becomes row
		B_shared[threadIdx.y][threadIdx.x] = (B_row < N && B_col < K) ? B[B_row * K + B_col] : 0.0F;

		__syncthreads();

		// Multiply the two tiles
		for (int i = 0; i < TILE_SIZE; ++i) {
			c_partial += A_shared[threadIdx.y][i] * B_shared[i][threadIdx.x];
		}

		__syncthreads();
	}

	// Write the result back to global memory
	if (row < M && col < N) {
		C[row * N + col] = c_partial + C[row * N + col];
	}
}

void cuda_test_gemm_atb(void) {
	int width = 128;
	int height = width;
	int channels = 32;
	if (width % 32 != 0) {
		printf("Input width must be a multiple of 32.\n");
		exit(EXIT_FAILURE);
	}

	size_t n_groups = 1;
	size_t n_filters = 128; // must be an even number
	size_t pad = 1;
	size_t stride = 1;
	size_t ksize = 3;
	size_t out_size = (width + 2 * pad - ksize) / stride + 1; // square image
	size_t out_n = ksize * ksize * channels * out_size * out_size;

	// M = # of filters
	// N = # of weights per filter (as if n_groups = 1) (ksize * ksize * input_channels)
	// K = # of outputs per filter (out_width * out_height)
	// A = M * K (dC/dz grads)
	// B = N * K -> transpose -> K * N
	// C = M * N
	size_t M = n_filters;
	size_t N = ksize * ksize * channels;
	size_t K = out_size * out_size;
	
	if (M % n_groups > 0 || N % n_groups > 0) {
		printf("Cannot divide filters or weights evenly between groups.\n");
		(void)getchar();
		exit(EXIT_FAILURE);
	}
	float* A = (float*)xcalloc(M * K, sizeof(float));
	float* B = (float*)xcalloc(N * K, sizeof(float));
	float* C = (float*)xcalloc(M * (N / n_groups), sizeof(float));

	fill_array_rand_float(A, M * K / n_groups, 0., 1.);
	fill_array_rand_float(B, N * K, 0., 1.);

	float* d_a = 0;
	float* d_b = 0;
	float* d_c = 0;

	CHECK_CUDA(cudaMalloc(&d_a, M * K / n_groups * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_b, N * K * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(float)));

	CHECK_CUDA(cudaMemcpy(d_a, A, M * K / n_groups * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_b, B, N * K * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemset(d_c, 0, M * N * sizeof(float)));

	M = M / n_groups; // # of filters per group
	N = N / n_groups; // # of weights per filter per group

	dim3 threads(TILE_SIZE, TILE_SIZE);
	dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	gemm_atb_kernel_shared KARGS(blocks, threads) (d_a, d_b, d_c, M, N, K);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("gemm_atb kernel execution time: %f ms\n", milliseconds);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	CHECK_CUDA(cudaGetLastError());

	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaMemcpy(C, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(d_a));
	CHECK_CUDA(cudaFree(d_b));
	CHECK_CUDA(cudaFree(d_c));


	float* gemm_atb_cpu = (float*)xcalloc(M * N, sizeof(float));
	gemm_atb_groups(M, N, K, A, B, gemm_atb_cpu, n_groups);
	free(A);
	free(B);
	float epsilon = 1e-5f;
	printf("Verifiying......\n");
	for (size_t i = 0; i < M * N; i++) {
		if (fabs(gemm_atb_cpu[i] - C[i]) > epsilon) {
			printf("Verification Failed: i = %zu, (gemm_atb_cpu)%f != (gemm_atb_gpu)%f\n", i, gemm_atb_cpu[i], C[i]);
			wait_for_key_then_exit();
		}
	}
	printf("Verifiction Success!!!\n");
}

__global__ void gemm_tab_kernel_shared(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int M, int N, int K)
{
	__shared__ float A_shared[TILE_SIZE][TILE_SIZE];
	__shared__ float B_shared[TILE_SIZE][TILE_SIZE];

	// These should map to the output C dimensions:
	int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // N rows
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // K columns

	float c_partial = 0.0F;

	for (int t = 0; t < (M + TILE_SIZE - 1) / TILE_SIZE; ++t) {
		// For A (transposed):
		int A_col = t * TILE_SIZE + threadIdx.x;  // M columns
		int A_row = row;                          // N rows
		A_shared[threadIdx.y][threadIdx.x] = (A_row < N && A_col < M) ? A[A_col * N + A_row] : 0.0F;

		// For B:
		int B_row = t * TILE_SIZE + threadIdx.y;  // M rows
		int B_col = col;                          // K columns
		B_shared[threadIdx.y][threadIdx.x] = (B_row < M && B_col < K) ? B[B_row * K + B_col] : 0.0F;

		__syncthreads();

		for (int i = 0; i < TILE_SIZE; ++i) {
			c_partial += A_shared[threadIdx.y][i] * B_shared[i][threadIdx.x];
		}

		__syncthreads();
	}

	if (row < N && col < K) {
		C[row * K + col] = c_partial + C[row * K + col];
	}
}

void cuda_test_gemm_tab(void) {
	int width = 128;
	int height = width;
	int channels = 32;
	if (width % 32 != 0) {
		printf("Input width must be a multiple of 32.\n");
		exit(EXIT_FAILURE);
	}

	size_t n_groups = 1;
	size_t n_filters = 32; // must be an even number
	size_t pad = 1;
	size_t stride = 1;
	size_t ksize = 3;
	size_t out_size = (width + 2 * pad - ksize) / stride + 1; // square image
	size_t out_n = ksize * ksize * channels * out_size * out_size;

	// M = # of filters
	// N = # of weights per filter (as if n_groups = 1)
	// K = # of outputs per filter
	// A = M * N -> transpose -> N * M (weights)
	// B = M * K (dC/dz grads)
	// C = N * K (col'd array to go through col2im)
	size_t M = n_filters;
	size_t N = ksize * ksize * channels;
	size_t K = out_size * out_size;
	
	if (M % n_groups > 0 || N % n_groups > 0) {
		printf("Cannot divide filters or weights evenly between groups.\n");
		(void)getchar();
		exit(EXIT_FAILURE);
	}
	float* A = (float*)xcalloc(M * (N / n_groups), sizeof(float));
	float* B = (float*)xcalloc(M * K, sizeof(float));
	float* C = (float*)xcalloc(N * K, sizeof(float));
	
	fill_array_rand_float(A, M * (N / n_groups), 0., 1.);
	fill_array_rand_float(B, M * K, 0., 1.);

	float* d_a = 0;
	float* d_b = 0;
	float* d_c = 0;

	CHECK_CUDA(cudaMalloc(&d_a, M * (N / n_groups) * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_b, M * K * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_c, N * K * sizeof(float)));

	CHECK_CUDA(cudaMemcpy(d_a, A, M * (N / n_groups) * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_b, B, M * K * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemset(d_c, 0, N * K * sizeof(float)));

	M = M / n_groups; // # of filters per group
	N = N / n_groups; // # of weights per filter per group

	dim3 threads(TILE_SIZE, TILE_SIZE);
	dim3 blocks((K + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	gemm_tab_kernel_shared KARGS(blocks, threads) (d_a, d_b, d_c, M, N, K);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("gemm_tab kernel execution time: %f ms\n", milliseconds);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	CHECK_CUDA(cudaGetLastError());

	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaMemcpy(C, d_c, sizeof(float) * N * K, cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(d_a));
	CHECK_CUDA(cudaFree(d_b));
	CHECK_CUDA(cudaFree(d_c));

	float* gemm_tab_cpu = (float*)xcalloc(N * K, sizeof(float));
	gemm_tab_groups(M, N, K, A, B, gemm_tab_cpu, n_groups);
	free(A);
	free(B);
	float epsilon = 1e-5f;
	printf("Verifiying......\n");
	for (size_t i = 0; i < N * K; i++) {
		if (fabs(gemm_tab_cpu[i] - C[i]) > epsilon) {
			printf("Verification Failed: i = %zu, (gemm_tab_cpu)%f != (gemm_tab_gpu)%f\n", i, gemm_tab_cpu[i], C[i]);
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