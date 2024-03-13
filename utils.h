#ifndef UTILS_H
#define UTILS_H


#ifdef __cplusplus
extern "C" {
#endif

#define UTILS_LOCATION __FILE__, __func__, __LINE__

int zz_str2int(char* string, const char* const filename, const char* const funcname, const int line);

#define str2int(str) zz_str2int(str, UTILS_LOCATION)


#ifdef _cplusplus
}
#endif
#endif