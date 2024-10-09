#ifndef LAYER_ROUTE_H
#define LAYER_ROUTE_H

#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

	void forward_route(layer* l, network* net);
	void backward_route(layer* l, network* net);

#ifdef __cplusplus
}
#endif
#endif