#ifndef UTILS_H
#define UTILS_H


#include <stdio.h>
#include "locmacro.h"
#include "list.h"


#define IMG_EXTS ".jpg,.jpeg,.bmp,.png"


#ifdef __cplusplus
extern "C" {
#endif

	int ___str2int(char* string, const char* const filename, const char* const funcname, const int line);
	float ___str2float(char* string, const char* const filename, const char* const funcname, const int line);
	size_t ___str2sizet(char* string, const char* const filename, const char* const funcname, const int line);

	#define str2int(str) ___str2int(str, NARDENET_LOCATION)
	#define str2float(str) ___str2float(str, NARDENET_LOCATION)
	#define str2sizet(str) ___str2sizet(str, NARDENET_LOCATION)

	int char_in_string(char c, char* str);
	float randu(float lower, float upper);
	double randn(double mean, double stddev);
	void get_random_numbers_no_repeats(size_t* arr, size_t size, size_t range_start, size_t range_end);
	char* read_line(FILE* file);
	int read_line_to_buff(FILE* file, char* buff, int buffsize);
	void clean_string(char* s);
	char** split_string(char* str, char* delimiters);
	void split_string_to_buff(char* str, char* delimiters, char** tokens);
	int file_exists(char* filename);
	FILE* get_filestream(char* filename, char* mode);
	void close_filestream(FILE* filestream);
	void lower_chars(char* s, size_t length);
	size_t get_line_count(FILE* file);
	list* get_files_list(char* dir, char* extensions);
	list* get_folders_list(char* dir, int include_path);
	void fix_dir_str(char* dir, size_t bufsize);
	void get_filename_from_path(char* dst, size_t dstsize, char* filepath, int remove_ext);
	int get_filename_ext_index(char* filename);
	/* arr & chars must be null terminated. index = index of arr to insert chars */
	int insert_chars(char* arr, size_t arr_size, int index, char* chars);
	size_t tokens_length(char** tokens);
	void print_str_array(char** strs, size_t count);
	void print_float_array(float* array, size_t size, char* text);
	void pprint_mat(float* data, int width, int height, int channels);
	void pprint_mat_batch(float* data, size_t width, size_t height, size_t channels, size_t batch_size);
	void wait_for_key_then_exit(void);
	void wait_for_key_then_continue(void);

#ifdef __cplusplus
}
#endif
#endif