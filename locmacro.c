#include "locmacro.h"
#include <stdio.h>


void print_location(const char* const filename, const char* const funcname, const int line) {
	printf("LOCATION: %s, %s, line %d", filename, funcname, line);
}
