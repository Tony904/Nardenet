#include "xallocs.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>


static void print_error_and_exit(const char* const filename, const char* const funcname, const int line);


void* custom_calloc(const size_t num_elements, size_t size_per_element, const char * const filename, const char * const funcname, const int line) {
    void* p = calloc(num_elements, size_per_element);
    if (!p) {
        fprintf(stderr, "Failed to calloc %d * %d bytes.\n", (int)num_elements, (int)size_per_element);
        print_error_and_exit(filename, funcname, line);
    }
    return p;
}

void* custom_malloc(const size_t num_bytes, const char * const filename, const char * const funcname, const int line) {
    void* p = malloc(num_bytes);
    if (!p) {
        fprintf(stderr, "Failed to malloc %d bytes.\n", (int)num_bytes);
        print_error_and_exit(filename, funcname, line);
    }
    return p;
}

void* custom_realloc(void* existing_mem, const size_t num_bytes_to_reallocate, const char * const filename, const char * const funcname, const int line) {
    void* p = realloc(existing_mem, num_bytes_to_reallocate);
    if (!p) {
        fprintf(stderr, "Failed to realloc %d bytes.", (int)num_bytes_to_reallocate);
        print_error_and_exit(filename, funcname, line);
    }
    return p;
}

static void print_error_and_exit(const char * const filename, const char * const funcname, const int line) {
#pragma warning(suppress:4996)
    fprintf(stderr, "Nardenet error location: %s, %s, line %d\nError Code %d: %s", filename, funcname, line, errno, strerror(errno));
    exit(EXIT_FAILURE);
}
