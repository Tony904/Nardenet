#ifndef NARDENET_H
#define NARDENET_H


#include <stdlib.h>
#include <stdio.h>


//#include "image.h"


#ifdef __cplusplus
extern "C" {
#endif

	// dll note: Importing nardenet.h into the header files of the source files that contain functions
	// to be exported allows me to not need to put __declspec(dllexport) in the function definition
	// in the source files. I only need to add __declspec(dllexport) in the declaration of exporting
	// functions in this file.

	// image.c
	//LIB_API int square(int x);


#ifdef __cplusplus
}
#endif
#endif
