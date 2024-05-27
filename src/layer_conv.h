#ifndef LAYER_CONV_H
#define LAYER_CONV_H


#include "network.h"


#ifdef __cplusplus
extern "C" {
#endif


	void forward_first(layer* l, network* net);
	void forward_conv(layer* l, network* net);
	void backprop_first(layer* l, network* net);
	void backprop_conv(layer* l, network* net);
	void activate_conv_relu(layer* l);
	void activate_conv_mish(layer* l);
	void activate_conv_logistic(layer* l);
	void activate_conv_none(layer* l);


#ifdef __cplusplus
}
#endif
#endif