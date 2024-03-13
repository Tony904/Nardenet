#ifndef XALLOCS_H
#define XALLOCS_H

#include <stdlib.h>  // provides access to size_t datatype
#include <stdio.h>



#ifdef __cplusplus
extern "C" {
#endif


#define XALLOCS_LOCATION __FILE__, __func__, __LINE__

void* zz_xcalloc(const size_t num_elements, const size_t size_per_element, const char * const filename, const char * const funcname, const int line);
void* zz_xmalloc(const size_t num_bytes, const char * filename, const char * const funcname, const int line);
void* zz_xrealloc(void* existing_mem, const size_t num_bytes_to_reallocate, const char * const filename, const char * const funcname, const int line);
void zz_xfree(void* ptr);

#define xcalloc(n, s) zz_xcalloc(n, s, XALLOCS_LOCATION)
#define xmalloc(ns) zz_xmalloc(ns, XALLOCS_LOCATION)
#define xrealloc(p, ns) zz_xrealloc(p, ns, XALLOCS_LOCATION)
#define xfree(p) zz_xfree(p)


#ifdef __cpluspls
}
#endif
#endif