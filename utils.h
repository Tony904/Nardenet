#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif


#define UTILS_LOCATION __FILE__, __func__, __LINE__

int zz_str2int(char* string, const char* const filename, const char* const funcname, const int line);
float zz_str2float(char* string, const char* const filename, const char* const funcname, const int line);
size_t zz_str2sizet(char* string, const char* const filename, const char* const funcname, const int line);

#define str2int(str) zz_str2int(str, UTILS_LOCATION)
#define str2float(str) zz_str2float(str, UTILS_LOCATION)
#define str2sizet(str) zz_str2sizet(str, UTILS_LOCATION)

typedef struct floatarr floatarr;
typedef struct intarr intarr;

typedef struct floatarr {
	size_t length;
	float* vals;
} floatarr;

typedef struct intarr {
	size_t length;
	int* vals;
} intarr;


#ifdef _cplusplus
}
#endif
#endif