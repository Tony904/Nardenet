#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include "xarrays.h"
#include "network.h"


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

int char_in_string(char c, char* str);
void print_lrpolicy(LR_POLICY lrp);
void print_layertype(LAYER_TYPE lt);
void print_activation(ACTIVATION a);
void print_floatarr(floatarr* p);
void free_floatarr(floatarr* farr);



#ifdef _cplusplus
}
#endif
#endif