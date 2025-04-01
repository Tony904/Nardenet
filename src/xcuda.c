#include "xcuda.h"
#include <stdio.h>
#include "utils.h"



void ___check_cuda(cudaError_t x, const char* const filename, const char* const funcname, const int line, const char* time) {
	if (x != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s\ntime: %s\n", cudaGetErrorString(x), time);
		print_location(filename, funcname, line);
		wait_for_key_then_exit();
	}
}

void print_gpu_props(void) {
    int n_devices;
    cudaGetDeviceCount(&n_devices);

    printf("Number of devices: %d\n", n_devices);

    for (int i = 0; i < n_devices; i++) {
        struct cudaDeviceProp prop;
        
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (MHz): %d\n",
            prop.memoryClockRate / 1024);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
            2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Total global memory (Gbytes) %.1f\n", (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
        printf("  Shared memory per block (Kbytes) %.1f\n", (float)(prop.sharedMemPerBlock) / 1024.0);
        printf("  minor-major: %d-%d\n", prop.minor, prop.major);
        printf("  Warp-size: %d\n", prop.warpSize);
        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("  Concurrent computation/communication: %s\n", prop.deviceOverlap ? "yes" : "no");
        printf("  Max threads/multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max blocks/multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("  Max threads/block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max grid size: %d, %d, %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  # of multiprocessors: %d\n", prop.multiProcessorCount);
        printf("\n");
    }
}