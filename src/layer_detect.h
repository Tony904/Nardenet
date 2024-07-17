#ifndef LAYER_DETECT_H
#define LAYER_DETECT_H

#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

	void forward_detect(layer* l, network* net);
	void backward_detect(layer* l, network* net);

#ifdef __cplusplus
}
#endif
#endif