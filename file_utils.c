#include "file_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int is_valid_fopen_mode(char* mode);
static void print_error_and_exit(const char* const filename);


FILE* get_file(char* filename, char* mode) {
#pragma warning(suppress:4996)
	FILE* file = fopen(filename, mode);
	if (file == 0) {
		print_error_and_exit(filename);
	}
	return file;
}

void close_filestream(FILE* filestream) {
	if (fclose(filestream) == EOF) {
		fprintf(stderr, "Error occured while closing a filestream. Continuing.");
	}
	filestream = NULL;
}

int is_valid_fopen_mode(char* mode) {
	char* modes[6] = { "r", "w", "a", "r+", "w+", "a+" };
	for (int i = 0; i < 6; i++) {
		if (strcmp(mode, modes[i]) == 0) return 1;
	}
	return 0;
}

static void print_error_and_exit(const char* const filename) {
#pragma warning(suppress:4996)
	fprintf(stderr, "Failed to open file: %s\nError Code %d: %s", filename, errno, strerror(errno));
	exit(EXIT_FAILURE);
}
