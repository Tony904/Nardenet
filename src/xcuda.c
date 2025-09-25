#include "xcuda.h"
#include <stdio.h>
#include <math.h>
#include "utils.h"



void gpu_not_defined(void) {
    printf("Cannot run GPU code. Install CUDA and compile Nardenet with preprocessor \"GPU\" defined.");
    wait_for_key_then_exit();
}

#ifdef GPU

void ___check_cuda(cudaError_t x, const char* const filename, const char* const funcname, const int line, const char* time) {
	if (x != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s\ntime: %s\n", cudaGetErrorString(x), time);
		print_location(filename, funcname, line);
		wait_for_key_then_exit();
	}
}

void ___cudaMalloc(void** devPtr, size_t size, const char* const filename, const char* const funcname, const int line, const char* time) {
    ___check_cuda(cudaMalloc(devPtr, size), filename, funcname, line, time);
}

void ___cudaMemcpy(void* dst, void* src, size_t size, enum cudaMemcpyKind kind, const char* const filename, const char* const funcname, const int line, const char* time) {
    ___check_cuda(cudaMemcpy(dst, src, size, kind), filename, funcname, line, time);
}

void print_gpu_float_array(float* gpu_array, size_t size, char* text) {
    float* buff = (float*)calloc(size, sizeof(float));
    if (!buff) {
        printf("calloc error");
        print_location(NARDENET_LOCATION);
        wait_for_key_then_exit();
    }
    CUDA_MEMCPY_D2H(buff, gpu_array, size * sizeof(float));
    printf("%s\n", text);
    printf("[GPU array]\n");
    for (size_t i = 0; i < size; i++) {
        printf("%f\n", buff[i]);
    }
    printf("[End GPU array]\n");
    free(buff);
}

void compare_cpu_gpu_arrays(float* cpu_array, float* gpu_array, size_t size, int layer_id, char* text) {
    float* buff = (float*)calloc(size, sizeof(float));
    if (!buff) {
        printf("calloc error");
        print_location(NARDENET_LOCATION);
        wait_for_key_then_exit();
    }
    CUDA_MEMCPY_D2H(buff, gpu_array, size * sizeof(float));

    float epsilon = 1e-5f;
    size_t zero_count = 0;
    for (size_t i = 0; i < size; i++) {
        //printf("%f =? %f\n", cpu_array[i], buff[i]);
        if (fabsf(cpu_array[i] - buff[i]) > epsilon) {
            printf("[CPU/GPU COMPARE - (layer id=%d) %s]\n", layer_id, text);
            printf("Large delta found: i = %zu, (cpu)%f, (gpu)%f\n", i, cpu_array[i], buff[i]);
            printf("zero count: %zu\r", zero_count);
            wait_for_key_then_continue();
        }
        if (cpu_array[i] == 0.0F && buff[i] == 0.0F) {
            zero_count++;
        }
    }
    free(buff);
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

#endif // GPU