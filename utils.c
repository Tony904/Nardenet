#include "utils.h"
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <inttypes.h>
#include <assert.h>


static void print_location_and_exit(const char* const filename, const char* const funcname, const int line);


int zz_str2int(char* str, const char* const filename, const char* const funcname, const int line) {
	errno = 0;
	char* p;
	int ret = (int)strtol(str, &p, 10);
	if (errno == ERANGE) {
		fprintf(stderr, "Error converting string (%s) to int via strtol().", str);
#pragma warning(suppress:4996)
		fprintf(stderr, "Error Code %d: %s", errno, strerror(errno));
		print_location_and_exit(filename, funcname, line);
	}
	if (p == str) {
		fprintf(stderr, "No number parsed in string (%s). String is empty or does not start with numbers.\n", str);
		print_location_and_exit(filename, funcname, line);
	}
	return ret;
}

size_t zz_str2sizet(char* str, const char* const filename, const char* const funcname, const int line) {
	errno = 0;
	char* p;
	size_t ret = (size_t)strtoumax(str, &p, 10);
	if (errno == ERANGE) {
		fprintf(stderr, "Error converting string (%s) to int via strtoumax().", str);
#pragma warning(suppress:4996)
		fprintf(stderr, "Error Code %d: %s", errno, strerror(errno));
		print_location_and_exit(filename, funcname, line);
	}
	if (p == str) {
		fprintf(stderr, "No number parsed in string (%s). String is empty or does not start with numbers.\n", str);
		print_location_and_exit(filename, funcname, line);
	}
	return ret;
}

float zz_str2float(char* str, const char* const filename, const char* const funcname, const int line) {
	errno = 0;
	char* p;
	float ret = strtof(str, &p);
	if (errno == ERANGE) {
		fprintf(stderr, "Error converting string (%s) to float via strtof().", str);
#pragma warning(suppress:4996)
		fprintf(stderr, "Error Code %d: %s", errno, strerror(errno));
		print_location_and_exit(filename, funcname, line);
	}
	if (p == str) {
		fprintf(stderr, "No number parsed in string (%s). String is empty or does not start with numbers.\n", str);
		print_location_and_exit(filename, funcname, line);
	}
	return ret;
}

int char_in_string(char c, char* str) {
	size_t length = strlen(str);
	size_t i;
	for (i = 0; i < length; i++) {
		if (c == str[i]) return 1;
	}
	return 0;
}

static void print_location_and_exit(const char* const filename, const char* const funcname, const int line) {
	fprintf(stderr, "Nardenet error location: %s, %s, line %d\n", filename, funcname, line);
	exit(EXIT_FAILURE);
}