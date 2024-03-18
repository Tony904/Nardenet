#ifndef FILE_UTILS_H
#define FILE_UTILS_H


#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif


FILE* get_file(char* filename, char* mode);
void close_filestream(FILE* filestream);


#ifdef __cplusplus
}
#endif
#endif