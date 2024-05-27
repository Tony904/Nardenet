#ifndef UTILS_H
#define UTILS_H


#include <stdio.h>
#include "locmacro.h"
#include "list.h"


#define IMG_EXTS ".jpg,.jpeg,.bmp,.png"


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
double randn(double mean, double stddev);
char* read_line(FILE* file);
int read_line_to_buff(FILE* file, char* buff, size_t buffsize);
void clean_string(char* s);
char** split_string(char* str, char* delimiters);
void split_string_to_buff(char* str, char* delimiters, char** tokens);
int file_exists(char* filename);
FILE* get_filestream(char* filename, char* mode);
void close_filestream(FILE* filestream);
size_t get_line_count(FILE* file);
list* get_files_list(char* dir, char* extensions);
size_t tokens_length(char** tokens);
void print_str_array(char** strs, size_t count);
void wait_for_key_then_exit(void);


#ifdef _cplusplus
}
#endif
#endif