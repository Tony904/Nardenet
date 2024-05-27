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