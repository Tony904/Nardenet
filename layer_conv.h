#ifndef LAYER_CONV_H
#define LAYER_CONV_H


#include "network.h"


#ifdef __cplusplus
extern "C" {
#endif


	void forward_layer_first(layer* l, network* net);
	void forward_layer_conv(layer* l, network* net);
	void backward_layer_first(layer* l, network* net);
	void backward_layer_conv(layer* l, network* net);


#ifdef __cplusplus
}
#endif
#endif