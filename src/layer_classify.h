#ifndef LAYER_CLASSIFY_H
#define LAYER_CLASSIFY_H


#include "network.h"


#ifdef __cplusplus
extern "C" {
#endif

	void forward_classify(layer* l, network* net);
	void backprop_classify(layer* l, network* net);

#ifdef __cplusplus
}
#endif
#endif