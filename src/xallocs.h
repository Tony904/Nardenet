#ifndef XALLOCS_H
#define XALLOCS_H


#include <stdlib.h>
#include "locmacro.h"


#ifdef __cplusplus
extern "C" {
#endif

	void* zz_xcalloc(const size_t num_elements, const size_t size_per_element, const char * const filename, const char * const funcname, const int line);
	void* zz_xmalloc(const size_t num_bytes, const char * filename, const char * const funcname, const int line);
	void* zz_xrealloc(void* existing_mem, const size_t new_num_bytes, const char * const filename, const char * const funcname, const int line);
	void zz_xfree(void* ptr, const char* const filename, const char* const funcname, const int line);

	#define xcalloc(n, s) zz_xcalloc(n, s, NARDENET_LOCATION)
	#define xmalloc(ns) zz_xmalloc(ns, NARDENET_LOCATION)
	#define xrealloc(p, ns) zz_xrealloc(p, ns, NARDENET_LOCATION)
	#define xfree(p) zz_xfree(p, NARDENET_LOCATION)

	void print_alloc_list(void);

#ifdef __cpluspls
}
#endif
#endif