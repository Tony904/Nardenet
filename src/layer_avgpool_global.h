#ifndef LAYER_AVGPOOL_GLOBAL_H
#define LAYER_AVGPOOL_GLOBAL_H

#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

	void forward_avgpool_global(layer* l, network* net);
	void backward_avgpool_global(layer* l, network* net);
	void forward_avgpool_global_gpu(layer* l, network* net);
	void backward_avgpool_global_gpu(layer* l, network* net);

#ifdef __cplusplus
}
#endif
#endif