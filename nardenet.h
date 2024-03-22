#ifndef NARDENET_API
#define NARDENET_API

#include <stdlib.h>
#if defined(_DEBUG) && !defined(_CRTDBG_MAP_ALLOC)
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new DEBUG_NEW
#endif

#include <stdio.h>


#ifndef LIB_API
#ifdef LIB_EXPORTS
#ifdef _MSC_VER  // _MSC_VER is defined when compiling with MS Visual Studio, also provides version info
#define LIB_API __declspec(dllexport)  // for windows
#else
#define LIB_API __attribute__((visibility("default")))  // for linux
#endif
#else
#define LIB_API
#endif
#endif


//#include "image.h"
//#include "xopencv.h"


#ifdef __cplusplus
extern "C" {
#endif


	//LIB_API image load_file_to_image();



#ifdef __cplusplus
}
#endif
#endif
