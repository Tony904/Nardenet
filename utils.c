#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <string.h>
#include <inttypes.h>
#include <assert.h>
#include <math.h>
#include "xallocs.h"


#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>  // unix-based systems
#endif


int is_valid_fopen_mode(char* mode);
static void print_location_and_exit(const char* const filename, const char* const funcname, const int line);
static void print_error_and_exit(const char* const filename);


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

double randn(double mean, double stddev) {
	static double n2 = 0.0;
	static int n2_cached = 0;
	if (n2_cached) {
		n2_cached = 0;
		return n2 * stddev + mean;
	}
	else {
		double x = 0.0;
		double y = 0.0;
		double r = 0.0;
		while (r == 0.0 || r > 1.0) {
			x = 2.0 * rand() / RAND_MAX - 1;
			y = 2.0 * rand() / RAND_MAX - 1;
			r = x * x + y * y;
		}
		double d = sqrt(-2.0 * log(r) / r);
		n2 = y * d;
		n2_cached = 1;
		return x * d * stddev + mean;
	}

}

/*
Allocates char array of size 512 and stores result of fgets.
Returns array on success.
Returns 0 if fgets failed.
*/
char* read_line(FILE* file) {
	int size = 512;
	char* line = (char*)xcalloc(size, sizeof(char));
	if (!fgets(line, size, file)) {  // fgets returns null pointer on fail or end-of-file + no chars read
		xfree(line);
		return 0;
	}
	return line;
}

/*
Removes whitespaces and line-end characters.
Removes comment character '#' and all characters after.
*/
void clean_string(char* str) {
	size_t length = strlen(str);
	size_t offset = 0;
	size_t i;
	char c;
	for (i = 0; i < length; i++) {
		c = str[i];
		if (c == '#') break;  // '#' is used for comments
		if (c == ' ' || c == '\n' || c == '\r') offset++;
		else str[i - offset] = c;
	}
	str[i - offset] = '\0';
}

/*
Splits string by delimiter and returns a null-terminated char* array with pointers to str.
Modifies str.
*/
char** split_string(char* str, char* delimiters) {
	size_t length = strlen(str);
	if (!length) return NULL;
	size_t i = 0;
	if (char_in_string(str[0], delimiters) || char_in_string(str[length - 1], delimiters)) {
		fprintf(stderr, "Line must not start or end with delimiter.\nDelimiters: %s\n Line: %s\n", delimiters, str);
		exit(EXIT_FAILURE);
	}
	size_t count = 1;
	for (i = 0; i < length; i++) {
		if (char_in_string(str[i], delimiters)) {
			if (char_in_string(str[i + 1], delimiters)) {
				fprintf(stderr, "Line must not contain consecutive delimiters.\nDelimiters: %s\n Line: %s\n", delimiters, str);
				exit(EXIT_FAILURE);
			}
			count++;
		}
	}
	char** strings = (char**)xcalloc(count + 1, sizeof(char*));
	strings[0] = &str[0];
	size_t j = 1;
	if (count > 1)
		for (i = 1; i < length; i++) {
			if (char_in_string(str[i], delimiters)) {
				str[i] = '\0';
				strings[j] = &str[i + 1];
				j++;
				i++;
			}
		}
	strings[count] = NULL;
	return strings;
}

int file_exists(char* filename) {
	FILE* file = fopen(filename, "r");
	if (file) {
		close_filestream(file);
		return 1;
	}
	printf("File %s does not exist.\n", filename);
	return 0;
}

FILE* get_filestream(char* filename, char* mode) {
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
}

int is_valid_fopen_mode(char* mode) {
	char* modes[6] = { "r", "w", "a", "r+", "w+", "a+" };
	for (int i = 0; i < 6; i++) {
		if (strcmp(mode, modes[i]) == 0) return 1;
	}
	return 0;
}

size_t get_line_count(FILE* file) {
	char buf[1024U];
	size_t counter = 0;
	while (1) {
		size_t n = fread((void*)buf, 1, 1024U, file);
		if (ferror(file)) print_location_and_exit(NARDENET_LOCATION);
		if (n == 0) return (size_t)0;
		size_t i;
		for (i = 0; i < n; i++) {
			if (buf[i] == '\n') counter++;
		}
		if (buf[i] != '\n') counter++;
		if (feof(file)) break;
	}
	rewind(file);
	return counter;
}

// extensions are a list of extensions separated by a comma. i.e. ".jpg,.jpeg,.bmp"
list* get_files_list(char* dir, char* extensions) {
	list* paths = new_list();
	size_t count = 0U;
	char buff[100];
	strcpy(buff, extensions);
	char** exts = split_string(buff, ",");
#ifdef _WIN32
	WIN32_FIND_DATA filedata;
	HANDLE handle;
	char search_path[MAX_PATH];
	snprintf(search_path, sizeof(search_path), "%s\\*.*", dir);
	handle = FindFirstFile(search_path, &filedata);
	if (handle == INVALID_HANDLE_VALUE) {
		if (GetLastError() == ERROR_NO_MORE_FILES) {
			printf("No files found with extension %s in directory %s\n", extensions, dir);
			xfree(paths);
			return (list*)0;
		}
		printf("Unexpected error occured while searching for first file in directory %s\n", dir);
		printf("Error Code: %d\n", GetLastError());
		wait_for_key_then_exit();
	}
	int ret = 1;
	while (1) {
		if (ret == 0) {
			if (GetLastError() == ERROR_NO_MORE_FILES) break;
			printf("Unexpected error occured while searching for files in directory %s\n", dir);
			printf("Error Code: %d\n", GetLastError());
			wait_for_key_then_exit();
		}
		if (!(filedata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
			char* ext1 = strrchr(filedata.cFileName, '.');
			if (!ext1) continue;
			for (size_t i = 0; i < tokens_length(exts); i++) {
				char* ext2 = strrchr(exts[i], '.');
				if (strcmp(ext1, ext2) == 0) {
					printf("%s\n", (char*)filedata.cFileName);
					size_t length = strlen(filedata.cFileName);
					char* path = (char*)xcalloc(length + 1, sizeof(char));
					strcpy(path, filedata.cFileName);
					list_append(paths, path);
					count++;
					break;
				}
			}
		}
		ret = (int)FindNextFile(handle, &filedata);
	}
	printf("# of files found: %zu\n", count);
	FindClose(handle);
	return paths;
#else  // Unix-based systems
	struct dirent* entry;
	DIR* dp = opendir(directory);
	if (dp == NULL) {
		perror("opendir");
		return;
	}
	while ((entry = readdir(dp))) {
		char* file_name = entry->d_name;
		char* ext1 = strrchr(file_name, '.');
		if (!ext1) continue;
		for (size_t i = 0; i < tokens_length(exts); i++) {
			char* ext2 = strrchr(exts[i], '.');
			if (strcmp(ext1, ext2) == 0) {
				printf("%s\n", (char*)file_name);
				size_t length = strlen(file_name);
				char* path = (char*)xcalloc(length + 1, sizeof(char));
				strcpy(path, file_name);
				list_append(paths, path);
				count++;
				break;
			}
		}
	}
	printf("# of files found: %zu\n", count);
	closedir(dp);
	return paths;
#endif
}

size_t tokens_length(char** tokens) {
	size_t i = 0;
	while (tokens[i] != NULL) {
		i++;
	}
	return i;
}

void wait_for_key_then_exit(void) {
	printf("\n\nPress ENTER to exit the program.");
	(void)getchar();
	exit(EXIT_FAILURE);
}

static void print_error_and_exit(const char* const filename) {
#pragma warning(suppress:4996)
	fprintf(stderr, "Failed to open file: %s\nError Code %d: %s", filename, errno, strerror(errno));
	exit(EXIT_FAILURE);
}

static void print_location_and_exit(const char* const filename, const char* const funcname, const int line) {
	fprintf(stderr, "Nardenet error location: %s, %s, line %d\n", filename, funcname, line);
	exit(EXIT_FAILURE);
}