#include "xcuda.h"
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <assert.h>
#include "utils.h"


#pragma warning(disable : 4996)

#define ARRSIZE 64  // size of char[] attributes in alloc_node struct

typedef struct alloc_node alloc_node;
typedef struct alloc_list alloc_list;

typedef struct alloc_node {
    void* p;
    size_t n_elements;
    size_t element_size;
    char filename[ARRSIZE];
    char funcname[ARRSIZE];
    int line;
    alloc_node* next;
    alloc_node* prev;
} alloc_node;

typedef struct alloc_list {
    int length;
    alloc_node* first;
    alloc_node* last;
} alloc_list;

static int track_allocs = 0;
static alloc_list allocs = { 0 };
static omp_lock_t allocs_lock;
static void print_location_and_exit(const char* const filename, const char* const funcname, const int line);

void initialize_allocs_lock(void);
int alloc_list_free_node(void* const p);
alloc_node* alloc_list_get_node(void* const p);
alloc_node* alloc_list_pop(void* const p);
void alloc_list_append(alloc_node* node);
alloc_node* new_alloc_node(void* const p, size_t n, size_t s, const char* const filename, const char* const funcname, const int line);
void alloc_node_set_location_fields(alloc_node* node, const char* const filename, const char* const funcname, const int line);

static cublasHandle_t cublas_handle;
static int cublas_handle_init = 0;


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

void ___check_cublas(cublasStatus_t x, const char* const filename, const char* const funcname, const int line, const char* time) {
    if (x != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS error: %s\ntime: %s\n", cudaGetErrorString(x), time);
        print_location(filename, funcname, line);
        wait_for_key_then_exit();
    }
}

void ___cudaMemcpy(void* dst, void* src, size_t size, enum cudaMemcpyKind kind, const char* const filename, const char* const funcname, const int line, const char* time) {
    ___check_cuda(cudaMemcpy(dst, src, size, kind), filename, funcname, line, time);
}

cublasHandle_t get_cublas_handle(void) {
    if (!cublas_handle_init) {
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
        cublas_handle_init = 1;
    }
    return cublas_handle;
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
    size_t span = 10;
    float epsilon = 1e-5f;
    size_t zero_count = 0;
    for (size_t i = 0; i < size; i++) {
        //printf("%f =? %f\n", cpu_array[i], buff[i]);
        if (fabsf(cpu_array[i] - buff[i]) > epsilon) {
            printf("[CPU/GPU COMPARE - (layer id=%d) %s]\n", layer_id, text);
            printf("Large delta found: i = %zu, (cpu)%f | %f(gpu)\n", i, cpu_array[i], buff[i]);
            printf("zero count: %zu\r", zero_count);

            size_t j = max(0, (int)i - (int)span);
            printf("[+-%zu elements around large delta] array size=%zu\n", span, size);
            for (; j <= i + span; j++) {
                if (j >= size) break;
                if (j == i) printf("\n[%zu] (cpu)%f | %f(gpu)\n\n", j, cpu_array[j], buff[j]);
                else printf("[%zu] (cpu)%f | %f(gpu)\n", j, cpu_array[j], buff[j]);
            }
            //wait_for_key_then_exit();
            wait_for_key_then_continue();
        }
        if (cpu_array[i] == 0.0F && buff[i] == 0.0F) {
            zero_count++;
        }
    }
    free(buff);
}

void print_cpu_gpu_arrays(float* cpu_array, float* gpu_array, size_t size, char* text) {
    float* buff = (float*)calloc(size, sizeof(float));
    if (!buff) {
        printf("calloc error");
        print_location(NARDENET_LOCATION);
        wait_for_key_then_exit();
    }
    CUDA_MEMCPY_D2H(buff, gpu_array, size * sizeof(float));
    printf("%s\n", text);
    for (size_t i = 0; i < size; i++) {
        printf("(cpu)%f | %f(gpu)\n", cpu_array[i], buff[i]);
    }
    free(buff);
    wait_for_key_then_continue();
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



void ___cudaMalloc(void** devPtr, const size_t num_elements, size_t size_per_element, const char* const filename, const char* const funcname, const int line, const char* time) {
    if (track_allocs) omp_set_lock(&allocs_lock);
    ___check_cuda(cudaMalloc(devPtr, num_elements * size_per_element), filename, funcname, line, time);
    if (!(*devPtr)) {
        fprintf(stderr, "Failed to cudaMalloc %zu * %zu bytes.\n", num_elements, size_per_element);
        print_location_and_exit(filename, funcname, line);
    }
    if (track_allocs) {
        alloc_node* node = new_alloc_node(*devPtr, num_elements, size_per_element, filename, funcname, line);
        alloc_list_append(node);
        omp_unset_lock(&allocs_lock);
    }
}

void ___cudaFree(void** devPtr, const char* const filename, const char* const funcname, const int line, const char* time) {
    if (!*devPtr) return;
    int do_free = 1;
    if (track_allocs) {
        omp_set_lock(&allocs_lock);
        do_free = alloc_list_free_node((void* const)*devPtr);
    }
    if (do_free) {
        ___check_cuda(cudaFree(*devPtr), filename, funcname, line, time);
        *devPtr = NULL;
    }
    if (track_allocs) omp_unset_lock(&allocs_lock);
}

#endif // GPU

void activate_cuda_alloc_tracking() {
    if (allocs.length != 0) {
        printf("Cannot enable cuda alloc tracking unless allocs.length is 0. (length = %d)\n", allocs.length);
        wait_for_key_then_exit();
    }
    initialize_allocs_lock();
}

void initialize_allocs_lock(void) {
    omp_init_lock(&allocs_lock);
}

#pragma warning(suppress: 4715)  // Not all control paths return a value. (because one exits the program)
alloc_node* new_alloc_node(void* const p, size_t n, size_t s, const char* const filename, const char* const funcname, const int line) {
    alloc_node* node = (alloc_node*)calloc(1, sizeof(alloc_node));
    if (!node) {
        fprintf(stderr, "(new_alloc_node, cuda) Failed to calloc %zu * %zu bytes.\n", n, s);
        print_location_and_exit(filename, funcname, line);
    }
    else {
        node->p = p;
        node->n_elements = n;
        node->element_size = s;
        alloc_node_set_location_fields(node, filename, funcname, line);
        return node;
    }
}

void alloc_node_set_location_fields(alloc_node* node, const char* const filename, const char* const funcname, const int line) {
    size_t start = max(0, strlen(filename) - ARRSIZE - 1);
    strcpy(node->filename, &filename[start]);
    start = max(0, strlen(funcname) - ARRSIZE - 1);
    strcpy(node->funcname, &funcname[start]);
    node->line = line;
}

void alloc_list_append(alloc_node* node) {
    if (allocs.length == 0) {
        allocs.first = node;
        allocs.last = node;
        allocs.length = 1;
        return;
    }
    node->prev = allocs.last;
    allocs.last->next = node;
    allocs.last = node;
    allocs.length++;
}

int alloc_list_free_node(void* const p) {
    alloc_node* node = alloc_list_pop(p);
    if (node) {
        free(node);
        return 1;
    }
    return 0;
}

alloc_node* alloc_list_pop(void* const p) {
    alloc_node* node = alloc_list_get_node(p);
    alloc_node* a = node->prev;
    alloc_node* b = node->next;
    if (!a) {  // popped node is first in list
        if (b) {
            allocs.first = b;
            b->prev = NULL;
        }
        else {  // popped node is only node in list
            allocs.first = NULL;
            allocs.last = NULL;
        }
    }
    else {
        if (b) {
            a->next = b;
            b->prev = a;
        }
        else {  // popped node is last in list
            allocs.last = a;
            a->next = NULL;
        }
    }
    allocs.length--;
    return node;
}

alloc_node* alloc_list_get_node(void* const p) {
    alloc_node* node = allocs.first;
    for (size_t i = 0; i < allocs.length; i++) {
        if (node->p == p) return node;
        node = node->next;
    }
    printf("\nError: allocs list node does not exist.\n");
    (void)getchar();
    exit(EXIT_FAILURE);
}

void print_cuda_alloc_list(void) {
    alloc_node* node = allocs.first;
    alloc_node n = { 0 };
    int i = 0;
    printf("\n\n[ALLOC LIST]\n");
    printf("alloc list length: %d\n\n", allocs.length);
    while (node) {
        n = *node;
        printf("[NODE %d]\n", i);
        printf("node address: %p\n", node);
        printf("p = %p\n", n.p);
        printf("bytes = %zu * %zu\n", n.n_elements, n.element_size);
        printf("filename = %s\n", n.filename);
        printf("funcname = %s\n", n.funcname);
        printf("line = %d\n", n.line);
        printf("prev node = %p\n", n.prev);
        printf("next node = %p\n\n", n.next);
        node = n.next;
        i++;
    }
    assert(i == allocs.length);
    printf("[END ALLOC LIST]\n");
}

