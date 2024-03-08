#ifndef XALLOCS_H
#define XALLOCS_H

#include <stdlib.h>  // provides access to size_t datatype
#include <stdio.h>



#ifdef __cplusplus
extern "C" {
#endif


#define LOCATION __FILE__, __func__, __LINE__

void* custom_calloc(const size_t num_elements, const size_t size_per_element, const char * const filename, const char * const funcname, const int line);
void* custom_malloc(const size_t num_bytes, const char * filename, const char * const funcname, const int line);
void* custom_realloc(void* existing_mem, const size_t num_bytes_to_reallocate, const char * const filename, const char * const funcname, const int line);

#define xcalloc(n, s) custom_calloc(n, s, LOCATION)
#define xmalloc(ns) custom_malloc(ns, LOCATION)
#define xrealloc(p, ns) custom_realloc(p, ns, LOCATION)


#ifdef __cpluspls
}
#endif
#endif