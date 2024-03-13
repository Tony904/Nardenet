#include "utils.h"
#include <stdlib.h>
#include <stdio.h>


static void print_error_and_exit(const char* const filename, const char* const funcname, const int line);


int zz_str2int(char* str, const char* const filename, const char* const funcname, const int line) {
	char* p;
	int ret = (int)strtol(str, &p, 10);
	if (ret == 0) {
		fprintf(stderr, "Error converting string (%s) to int.", str);
		print_error_and_exit(filename, funcname, line);
	}
	return ret;
}

static void print_error_and_exit(const char* const filename, const char* const funcname, const int line) {
#pragma warning(suppress:4996)
	fprintf(stderr, "Nardenet error location: %s, %s, line %d\n", filename, funcname, line);
	exit(EXIT_FAILURE);
}