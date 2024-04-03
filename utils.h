#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include "locmacro.h"



#ifdef __cplusplus
extern "C" {
#endif


int zz_str2int(char* string, const char* const filename, const char* const funcname, const int line);
float zz_str2float(char* string, const char* const filename, const char* const funcname, const int line);
size_t zz_str2sizet(char* string, const char* const filename, const char* const funcname, const int line);

#define str2int(str) zz_str2int(str, NARDENET_LOCATION)
#define str2float(str) zz_str2float(str, NARDENET_LOCATION)
#define str2sizet(str) zz_str2sizet(str, NARDENET_LOCATION)

int char_in_string(char c, char* str);


#ifdef _cplusplus
}
#endif
#endif