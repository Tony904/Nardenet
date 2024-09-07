#ifndef STATE_H
#define STATE_H


#include <stdlib.h>
#include "network.h"


#ifdef __cplusplus
extern "C" {
#endif

	void save_state(network* net);
	void load_state(network* net);
	/*void test_save_state(void);
	void test_load_state(void);*/

#ifdef __cplusplus
}
#endif
#endif