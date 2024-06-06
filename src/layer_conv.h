#ifndef LAYER_CONV_H
#define LAYER_CONV_H


#include "network.h"


#ifdef __cplusplus
extern "C" {
#endif

	void forward_conv(layer* l);
	void backprop_conv(layer* l, network* net);

#ifdef __cplusplus
}
#endif
#endif