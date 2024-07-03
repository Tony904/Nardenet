#ifndef LAYER_MAXPOOL_H
#define LAYER_MAXPOOL_H


#include "network.h"


#ifdef __cplusplus
extern "C" {
#endif

	void forward_maxpool(layer* l, network* net);
	void backward_maxpool(layer* l, network* net);
	void test_forward_maxpool(void);

#ifdef __cplusplus
}
#endif
#endif