#include "xallocs.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <omp.h>



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

static alloc_list allocs = { 0 };
static omp_lock_t xallocs_lock;
static void print_location_and_exit(const char* const filename, const char* const funcname, const int line);

void alloc_list_free_node(void* const p);
alloc_node* alloc_list_get_node(void* const p);
alloc_node* alloc_list_pop(void* const p);
void alloc_list_append(alloc_node* node);
alloc_node* new_alloc_node(void* const p, size_t n, size_t s, const char* const filename, const char* const funcname, const int line);
void alloc_node_set_location_members(alloc_node* node, const char* const filename, const char* const funcname, const int line);



#pragma warning (suppress: 4715)  // Not all control paths return a value. (because one exits the program)
void* ___xcalloc(const size_t num_elements, size_t size_per_element, const char * const filename, const char * const funcname, const int line) {
    omp_set_lock(&xallocs_lock);
    void* p = calloc(num_elements, size_per_element);
    if (!p) {
        fprintf(stderr, "Failed to calloc %zu * %zu bytes.\n", num_elements, size_per_element);
        print_location_and_exit(filename, funcname, line);
    }
    alloc_node* node = new_alloc_node((void* const)p, num_elements, size_per_element, filename, funcname, line);
    alloc_list_append(node);
    omp_unset_lock(&xallocs_lock);
    return p;
}

void* ___xmalloc(const size_t num_bytes, const char * const filename, const char * const funcname, const int line) {
    omp_set_lock(&xallocs_lock);
    void* p = malloc(num_bytes);
    if (!p) {
        fprintf(stderr, "Failed to malloc %zu bytes.\n", num_bytes);
        print_location_and_exit(filename, funcname, line);
    }
    alloc_node* node = new_alloc_node((void* const)p, 1, num_bytes, filename, funcname, line);
    alloc_list_append(node);
    omp_unset_lock(&xallocs_lock);
    return p;
}

void* ___xrealloc(void* existing_mem, const size_t new_num_bytes, const char * const filename, const char * const funcname, const int line) {
    omp_set_lock(&xallocs_lock);
    void* x = existing_mem;
    void* p = realloc(existing_mem, new_num_bytes);
    if (!p) {
        fprintf(stderr, "Failed to realloc %zu bytes.", new_num_bytes);
        print_location_and_exit(filename, funcname, line);
    }
    alloc_node* node = alloc_list_get_node(x);
    node->p = p;
    node->n_elements = 1;
    node->element_size = new_num_bytes;
    alloc_node_set_location_members(node, filename, funcname, line);
    omp_unset_lock(&xallocs_lock);
    return p;
}

#pragma warning(suppress: 4100)  // C4100: Unused parameter(s).
void ___xfree(void* ptr, const char* const filename, const char* const funcname, const int line) {
    omp_set_lock(&xallocs_lock);
    //print_location_and_exit(filename, funcname, line);
    if (!ptr) {
        omp_unset_lock(&xallocs_lock); 
        return;
    }
    alloc_list_free_node((void* const)ptr);
    free(ptr);
    omp_unset_lock(&xallocs_lock);
}

static void print_location_and_exit(const char * const filename, const char * const funcname, const int line) {
#pragma warning(suppress:4996)  // C4996: Use of deprecated function, variable, or typedef. (strerror)
    fprintf(stderr, "Nardenet error location: %s, %s, line %d\nError Code %d: %s", filename, funcname, line, errno, strerror(errno));
    printf("\n\nPress ENTER to exit the program.");
    (void)getchar();
    exit(EXIT_FAILURE);
}

void alloc_list_free_node(void* const p) {
    alloc_node* node = alloc_list_pop(p);
    if (node) free(node);
}

alloc_node* alloc_list_get_node(void* const p) {
    alloc_node* node = allocs.first;
    for (size_t i = 0; i < allocs.length; i++) {
        if (node->p == p) return node;
        node = node->next;
    }
    printf("\nError: allocs list node does not exit.\n");
    (void)getchar();
    exit(EXIT_FAILURE);
}

alloc_node* alloc_list_pop(void* const p) {
    alloc_node* node = alloc_list_get_node(p);
    if (!node) return NULL;
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

#pragma warning(suppress: 4715)  // Not all control paths return a value. (because one exits the program)
alloc_node* new_alloc_node(void* const p, size_t n, size_t s, const char* const filename, const char* const funcname, const int line) {
    alloc_node* node = (alloc_node*)calloc(1, sizeof(alloc_node));
    if (node == NULL) {
        fprintf(stderr, "(allocs_append) Failed to calloc %zu * %zu bytes.\n", n, s);
        print_location_and_exit(filename, funcname, line);
    }
    else {
        node->p = p;
        node->n_elements = n;
        node->element_size = s;
        alloc_node_set_location_members(node, filename, funcname, line);
        return node;
    }
}

void alloc_node_set_location_members(alloc_node* node, const char* const filename, const char* const funcname, const int line) {
    size_t length = strlen(filename);
    size_t start = 0;
    if (length > ARRSIZE - 1) start = length - ARRSIZE - 1;
    strcpy(node->filename, &filename[start]);
    length = strlen(funcname);
    start = 0;
    if (length > ARRSIZE - 1) start = length - ARRSIZE - 1;
    strcpy(node->funcname, &funcname[start]);
    node->line = line;
}

void initialize_xallocs_lock(void) {
    omp_init_lock(&xallocs_lock);
}

void print_alloc_list(void) {
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

