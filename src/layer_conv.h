#ifndef LAYER_CONV_H
#define LAYER_CONV_H


#include "network.h"


#ifdef __cplusplus
extern "C" {
#endif

	void test_forward_conv(void);
	void forward_conv(layer* l, network* net);
	void backward_conv(layer* l, network* net);
	void update_conv(layer* l, network* net);

#ifdef __cplusplus
}
#endif
#endif