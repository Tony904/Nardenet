#ifndef LAYER_CLASSIFY_H
#define LAYER_CLASSIFY_H


#include "network.h"


#ifdef __cplusplus
extern "C" {
#endif

	void forward_layer_classify(layer* l, network* net);
	void backward_layer_classify(layer* l, network* net);
	void activate_classify(layer* l);


#ifdef __cplusplus
}
#endif
#endif