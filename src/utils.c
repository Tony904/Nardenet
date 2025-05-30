#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <string.h>
#include <inttypes.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <ctype.h>
#include "xallocs.h"


// Need this cus Microsoft says so
#define MICROSOFT_WINDOWS_WINBASE_H_DEFINE_INTERLOCKED_CPLUSPLUS_OVERLOADS 0


#define IS_WIN defined(_WIN32) || defined(_WIN64)
#ifdef IS_WIN
#include <windows.h>
#else
#include <dirent.h>  // unix-based systems
#include <sys/stat.h>
#endif


#ifdef IS_WIN
void str2wstr(const char* str, wchar_t* buf);
void wstr2str(const wchar_t* wstr, char* buf);
#endif


int is_valid_fopen_mode(char* mode);
static void print_location_and_exit(const char* const filename, const char* const funcname, const int line);
static void print_error_and_exit(const char* const filename);


void fill_array(float* arr, size_t size, float val) {
	size_t i;
#pragma omp parallel for
	for (i = 0; i < size; i++) {
		arr[i] = val;
	}
}

void zero_array(float* arr, size_t size) {
	size_t i;
#pragma omp parallel for
	for (i = 0; i < size; i++) {
		arr[i] = 0.0F;
	}
}

void fill_array_increment(float* arr, size_t size, float start_val, float increment) {
	size_t i;
#pragma omp parallel for firstprivate(start_val, increment)
	for (i = 0; i < size; i++) {
		arr[i] = start_val + (float)i * increment;
	}
}

void fill_array_rand_float(float* arr, size_t size, double mean, double sdev) {
	size_t i;
#pragma omp parallel for
	for (i = 0; i < size; i++) {
		arr[i] = randn(mean, sdev);
	}
}

float sum_array(float* arr, size_t size) {
	float sum = 0.0F;
	for (size_t i = 0; i < size; i++) {
		sum += arr[i];
	}
	return sum;
}

int char_in_string(char c, char* str) {
	size_t length = strlen(str);
	size_t i;
	for (i = 0; i < length; i++) {
		if (c == str[i]) return 1;
	}
	return 0;
}

float randu(float lower, float upper) {
	float scale = rand() / (float)RAND_MAX;
	return lower + scale * (upper - lower);
}

double randn(double mean, double stddev) {
	static double n2 = 0.0;
	static int n2_cached = 0;
	if (n2_cached) {
		n2_cached = 0;
		return n2 * stddev + mean;
	}
	double x = 0.0;
	double y = 0.0;
	double r = 0.0;
	while (r == 0.0 || r > 1.0) {
		x = 2.0 * rand() / RAND_MAX - 1.0;
		y = 2.0 * rand() / RAND_MAX - 1.0;
		r = x * x + y * y;
	}
	double d = sqrt(-2.0 * log(r) / r);
	n2 = y * d;
	n2_cached = 1;
	return x * d * stddev + mean;
}

// Inclusive range_start and range_end
void get_random_numbers_no_repeats(size_t* arr, size_t size, size_t range_start, size_t range_end) {
	size_t range = range_end - range_start + 1;
	if (size != range) {
		printf("Error: Size of array (%zu) is not equal to range. (%zu)\n", size, range);
		wait_for_key_then_exit();
	}
	if (!range) {
		printf("Error: range equals zero.\n");
		wait_for_key_then_exit();
	}
	for (size_t i = 0; i < range; i++) arr[i] = range_start + i;
	// Fisher-Yates shuffle
	for (size_t i = range - 1; i > 0; i--) {
		size_t j = rand() % (i + 1);
		size_t temp = arr[i];
		arr[i] = arr[j];
		arr[j] = temp;
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

int read_line_to_buff(FILE* file, char* buff, int buffsize) {
	if (!fgets(buff, buffsize, file)) return 0;
	return 1;
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
	if (char_in_string(str[0], delimiters) || char_in_string(str[length - 1], delimiters)) {
		fprintf(stderr, "Line must not start or end with delimiter.\nDelimiters: %s\n Line: %s\n", delimiters, str);
		exit(EXIT_FAILURE);
	}
	size_t i = 0;
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
	strings[0] = str;
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

void split_string_to_buff(char* str, char* delimiters, char** tokens) {
	size_t length = strlen(str);
	if (!length) {
		tokens[0] = NULL;
		return;
	}
	if (char_in_string(str[0], delimiters) || char_in_string(str[length - 1], delimiters)) {
		fprintf(stderr, "Line must not start or end with delimiter.\nDelimiters: %s\n Line: %s\n", delimiters, str);
		exit(EXIT_FAILURE);
	}
	size_t i = 0;
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
	tokens[0] = str;
	size_t j = 1;
	if (count > 1)
		for (i = 1; i < length; i++) {
			if (char_in_string(str[i], delimiters)) {
				str[i] = '\0';
				tokens[j] = &str[i + 1];
				j++;
				i++;
			}
		}
	tokens[count] = NULL;
}

int file_exists(char* filename) {
#ifdef IS_WIN
	DWORD fileattr = GetFileAttributesA(filename);
	if (fileattr == INVALID_FILE_ATTRIBUTES) {
		printf("File %s does not exist.\n", filename);
		return 0;
	}
	return 1;
#else
	struct stat buffer;
	int result = (stat(filename, &buffer) == 0);
	if (result == 0) {
		printf("File %s does not exist.\n", filename);
		return 0;
	}
	return result;
#endif
	/*FILE* file = fopen(filename, "r");
	if (file) {
		close_filestream(file);
		return 1;
	}
	printf("File %s does not exist.\n", filename);
	return 0;*/
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

void lower_chars(char* s, size_t length) {
	for (size_t i = 0; i < length; i++) s[i] = (char)tolower(s[i]);
}

size_t get_line_count(FILE* file) {
	char buf[1024] = { 0 };
	size_t counter = 0;
	while (1) {
		size_t n = fread((void*)buf, 1, 1024, file);
		if (ferror(file)) print_location_and_exit(NARDENET_LOCATION);
		if (n == 0) return (int)0;
		char last = buf[n - 1];
		for (size_t i = 0; i < n; i++) {
			if (buf[i] == '\n') counter++;
			buf[i] = '0';
		}
		if (last != '\n' && n != 1024) counter++;
		if (feof(file)) break;
	}
	rewind(file);
	return counter;
}

#ifdef IS_WIN
void str2wstr(const char* str, wchar_t* buf) {
	size_t length = strlen(str) + 1;
	mbstowcs(buf, str, length);
}

void wstr2str(const wchar_t* wstr, char* buf) {
	size_t length = wcslen(wstr) + 1;
	size_t converted = 0;
	wcstombs_s(&converted, buf, length, wstr, length - 1);
}
#endif

/* extensions are a list of extensions separated by a comma.i.e. ".jpg,.jpeg,.bmp" 
NOTE: All allocations are done with calloc, not xcalloc. Do not free with xfree(). */
list* get_files_list(char* dir, char* extensions) {
	list* paths = (list*)calloc(1, sizeof(list));
	if (!paths) exit(EXIT_FAILURE);
	size_t count = 0;
	char buff[100];
	strcpy(buff, extensions);
	char* exts[100];
	split_string_to_buff(buff, ",", exts);
#ifdef IS_WIN
	WIN32_FIND_DATA filedata;
	HANDLE handle;
	char search_path[MAX_PATH] = { 0 };
	snprintf(search_path, sizeof(search_path), "%s*.*", dir);
	wchar_t wspath[MAX_PATH] = { 0 };
	str2wstr(search_path, wspath);
	handle = FindFirstFile(wspath, &filedata);
	if (handle == INVALID_HANDLE_VALUE) {
		if (GetLastError() == ERROR_NO_MORE_FILES) {
			printf("No files found with extension %s in directory %s\n", extensions, dir);
			free(paths);
			return (list*)0;
		}
		printf("Unexpected error occured while searching for first file in directory %s\nDirectory may be empty or does not exist.", dir);
		printf("\nError Code: %d\n", GetLastError());
		wait_for_key_then_exit();
	}
	int ret = 1;
	char fullpath[MAX_PATH] = { 0 };
	char filename[MAX_PATH] = { 0 };
	while (1) {
		if (ret == 0) {
			if (GetLastError() == ERROR_NO_MORE_FILES) break;
			printf("Unexpected error occured while searching for files in directory %s\nDirectory may be empty or does not exist.", dir);
			printf("\nError Code: %d\n", GetLastError());
			wait_for_key_then_exit();
		}
		if (!(filedata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
			//char* filename = wstr2str(filedata.cFileName);
			wstr2str(filedata.cFileName, filename);
			char* ext = strrchr(filename, '.');
			lower_chars(ext, strlen(ext));
			if (ext) {
				for (size_t i = 0; i < tokens_length(exts); i++) {
					if (strcmp(ext, exts[i]) == 0) {
						snprintf(fullpath, sizeof(fullpath), "%s%s", dir, filename);
						size_t length = strlen(fullpath);
						char* path = (char*)calloc(length + 1, sizeof(char));
						if (!path) exit(EXIT_FAILURE);
						strcpy(path, fullpath);
						list_append_calloc(paths, path);
						count++;
						break;
					}
				}
			}
		}
		ret = (int)FindNextFile(handle, &filedata);
	}
	//printf("# of files found in %s:\n%zu\n", dir, count);
	FindClose(handle);
#else  // Unix-based systems
	struct dirent* entry;
	DIR* dp = opendir(directory);
	if (dp == NULL) {
		perror("opendir");
		return;
	}
	while (entry = readdir(dp)) {
		char* filename = entry->d_name;
		char* ext1 = strrchr(filename, '.');
		if (!ext1) continue;
		for (size_t i = 0; i < tokens_length(exts); i++) {
			char* ext2 = strrchr(exts[i], '.');
			if (strcmp(ext1, ext2) == 0) {
				printf("%s\n", (char*)filename);
				size_t length = strlen(filename);
				char* path = (char*)calloc(length + 1, sizeof(char));
				if (!path) exit(EXIT_FAILURE);
				strcpy(path, filename);
				list_append_calloc(paths, path);
				count++;
				break;
			}
		}
	}
	printf("# of files found: %zu\n", count);
	closedir(dp);
#endif
	return paths;
}

list* get_folders_list(char* dir, int include_path) {
	list* paths = (list*)calloc(1, sizeof(list));
	if (!paths) exit(EXIT_FAILURE);
	size_t count = 0;
	char fullpath[MAX_PATH] = { 0 };
#ifdef IS_WIN
	WIN32_FIND_DATA filedata;
	HANDLE handle;
	char search_path[MAX_PATH] = { 0 };
	snprintf(search_path, sizeof(search_path), "%s*.*", dir);
	wchar_t wspath[MAX_PATH] = { 0 };
	str2wstr(search_path, wspath);
	handle = FindFirstFile(wspath, &filedata);
	if (handle == INVALID_HANDLE_VALUE) {
		if (GetLastError() == ERROR_NO_MORE_FILES) {
			printf("No items found in directory %s\n", dir);
			free(paths);
			return (list*)0;
		}
		printf("Unexpected error occured while searching for first item in directory %s\nDirectory may be empty or does not exist.", dir);
		printf("\nError Code: %d\n", GetLastError());
		wait_for_key_then_exit();
	}
	char foldername[MAX_PATH] = { 0 };
	int ret = 1;
	while (1) {
		if (ret == 0) {
			if (GetLastError() == ERROR_NO_MORE_FILES) break;
			printf("Unexpected error occured while searching for files in directory %s\nDirectory may be empty or does not exist.", dir);
			printf("\nError Code: %d\n", GetLastError());
			wait_for_key_then_exit();
		}
		if (filedata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
			//char* foldername = wstr2str(filedata.cFileName);
			wstr2str(filedata.cFileName, foldername);
			if (strcmp(foldername, ".") != 0 && strcmp(foldername, "..") != 0) {
				if (include_path) snprintf(fullpath, sizeof(fullpath), "%s%s", dir, foldername);
				else strcpy(fullpath, foldername);
				size_t length = strlen(fullpath);
				char* path = (char*)calloc(length + 1, sizeof(char));
				if (!path) exit(EXIT_FAILURE);
				strcpy(path, fullpath);
				list_append_calloc(paths, path);
				count++;
			}
		}
		ret = (int)FindNextFile(handle, &filedata);
	}
	printf("# of folders found: %zu\n", count);
	FindClose(handle);
#else  // Unix-based systems
	struct dirent* entry;
	DIR* dp = opendir(dir);
	if (dp == NULL) {
		perror("opendir");
		return;
	}
	while (entry = readdir(dp)) {
		if (entry->d_type != DT_DIR) continue;
		if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
		char* foldername = (char*)entry->d_name;
		if (include_path) snprintf(fullpath, sizeof(fullpath), "%s%s", dir, foldername);
		else strcpy(fullpath, foldername);
		size_t length = strlen(foldername);
		char* path = (char*)xcalloc(length + 1, sizeof(char));
		strcpy(path, foldername);
		list_append(paths, path);
		count++;
	}
	printf("# of folders found: %zu\n", count);
	closedir(dp);
#endif
	return paths;
}

void fix_dir_str(char* dir, size_t bufsize) {
	size_t length = strlen(dir);
	if (dir[length - 1] == '\\') return;
	if (bufsize > length + 1) {
		dir[length] = '\\';
		dir[length + 1] = 0;
	}
}

void get_filename_from_path(char* dst, size_t dstsize, char* filepath, int remove_ext) {
	size_t length = strlen(filepath);
	size_t i;
	for (i = length; i; i--) {
		if (filepath[i] == '\\') break;
	}
	char* s = &filepath[i + 1];
	length = strlen(s);
	if (remove_ext) {
		size_t j;
		for (j = length; j; j--) {
			if (s[j] == '.') {
				break;
			}
		}
		if (dstsize < j) {
			printf("File name is too long: %s\nMust be less than %zu characters without extension.\n", s, dstsize);
			wait_for_key_then_exit();
		}
		memcpy(dst, s, j);
	}
	if (dstsize < length) {
		printf("File name is too long: %s\nMust be less than %zu characters.\n", s, dstsize);
		wait_for_key_then_exit();
	}
	strcpy(dst, s);
}

/* Returns index of the last '.' character in filename.
   If result is -1 that means no '.' was found. */
int get_filename_ext_index(char* filename) {
	int length = (int)strlen(filename);
	for (int ii = length; ii; ii--) {
		int i = ii - 1;
		if (filename[i] == '.') return i;
	}
	return -1;
}

/* index = index of arr to insert chars */
int insert_chars(char* arr, size_t arr_size, int index, char* chars) {
	size_t L1 = strlen(arr);
	size_t L2 = strlen(chars);
	if (L1 + L2 > arr_size) {
		printf("Array size too small to insert chars: \"%s\"", chars);
		return 0;
	}
	if (L1 + L2 > 1024) {
		printf("New char array must not be greater than 1024, is %zu\n.", L1 + L2);
		return 0;
	}
	char buf[1024] = { 0 };
	char* x = &arr[index];
	char* y = &arr[index + L2];
	strcpy(buf, x);
	strcpy(y, buf);
	memcpy(x, chars, L2);
	return 1;
}

size_t tokens_length(char** tokens) {
	size_t i = 0;
	while (tokens[i] != NULL) {
		i++;
	}
	return i;
}

void print_str_array(char** strs, size_t count) {
	for (size_t i = 0; i < count; i++) {
		printf("%s\n", strs[i]);
	}
}

void print_float_array(float* array, size_t size) {
	for (size_t i = 0; i < size; i++) {
		printf("%f\n", array[i]);
	}
}

void pprint_mat(float* data, int width, int height, int channels) {
	printf("\nMATRIX");
	for (int channel = 0; channel < channels; channel++) {
		for (int row = 0; row < height; row++) {
			printf("\n");
			for (int col = 0; col < width; col++) {
				float val = data[channel * width * height + row * width + col];
				if (val < 10 && val >= 0) printf("%0.1f   ", val);
				else if (val >= 10 && val < 100) printf("%0.1f  ", val);
				else printf("%0.1f ", val);
			}
		}
		printf("(ch%d)", channel);
	}
	printf("\nend\n\n");
}

void pprint_mat_batch(float* data, size_t width, size_t height, size_t channels, size_t batch_size) {
	size_t n = width * height * channels;
	printf("\nMATRIX");
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t channel = 0; channel < channels; channel++) {
			for (size_t row = 0; row < height; row++) {
				printf("\n");
				for (size_t col = 0; col < width; col++) {
					float val = data[b * n + channel * width * height + row * width + col];
					if (val < 10 && val >= 0) printf("%0.1f   ", val);
					else if (val >= 10 && val < 100) printf("%0.1f  ", val);
					else printf("%0.1f ", val);
				}
			}
			printf("(ch %zu)", channel);
		}
		printf("(batch %zu)", b);
	}
	printf("\nend\n\n");
}

void wait_for_key_then_exit(void) {
	printf("\n\nPress ENTER to exit the program.");
	(void)getchar();
	exit(EXIT_FAILURE);
}

static void print_error_and_exit(const char* const filename) {
#pragma warning(suppress:4996)
	fprintf(stderr, "Failed to open file: %s\nError Code %d: %s\n", filename, errno, strerror(errno));
	printf("Press ENTER to exit the program.");
	(void)getchar();
	exit(EXIT_FAILURE);
}

static void print_location_and_exit(const char* const filename, const char* const funcname, const int line) {
	fprintf(stderr, "Nardenet error location: %s, %s, line %d\n", filename, funcname, line);
	printf("Press ENTER to exit the program.");
	(void)getchar();
	exit(EXIT_FAILURE);
}

int ___str2int(char* str, const char* const filename, const char* const funcname, const int line) {
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

size_t ___str2sizet(char* str, const char* const filename, const char* const funcname, const int line) {
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

float ___str2float(char* str, const char* const filename, const char* const funcname, const int line) {
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