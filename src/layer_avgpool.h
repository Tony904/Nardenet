#ifndef LAYER_AVGPOOL_H
#define LAYER_AVGPOOL_H

#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

	void forward_avgpool(layer* l, network* net);
	void backward_avgpool(layer* l, network* net);

#ifdef __cplusplus
}
#endif
#endif