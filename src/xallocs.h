#ifndef XALLOCS_H
#define XALLOCS_H


#include <stdlib.h>
#include "locmacro.h"


#ifdef __cplusplus
extern "C" {
#endif

	void* ___xcalloc(const size_t num_elements, const size_t size_per_element, const char * const filename, const char * const funcname, const int line);
	void* ___xmalloc(const size_t num_bytes, const char * filename, const char * const funcname, const int line);
	void* ___xrealloc(void* existing_mem, const size_t new_num_bytes, const char * const filename, const char * const funcname, const int line);
	void ___xfree(void* ptr, const char* const filename, const char* const funcname, const int line);
	void activate_xalloc_tracking(void);

	#define xcalloc(n, s) ___xcalloc(n, s, NARDENET_LOCATION)
	#define xmalloc(ns) ___xmalloc(ns, NARDENET_LOCATION)
	#define xrealloc(p, ns) ___xrealloc(p, ns, NARDENET_LOCATION)
	#define xfree(p) ___xfree(p, NARDENET_LOCATION)

	void print_alloc_list(void);

#ifdef __cplusplus
}
#endif
#endif