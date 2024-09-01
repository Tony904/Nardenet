#ifndef STATE_H
#define STATE_H


#include <stdlib.h>
#include "network.h"


#ifdef __cplusplus
extern "C" {
#endif

	void save_state(network* net);
	void load_state(network* net);

#ifdef __cplusplus
}
#endif
#endif