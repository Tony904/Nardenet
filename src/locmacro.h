#ifndef LOCMACRO_H
#define LOCMACRO_H


#ifdef __cplusplus
extern "C" {
#endif


void print_location(const char* const filename, const char* const funcname, const int line);

#define NARDENET_LOCATION __FILE__, __func__, __LINE__


#ifdef __cplusplus
}
#endif
#endif