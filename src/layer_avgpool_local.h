#ifndef LAYER_AVGPOOL_LOCAL_H
#define LAYER_AVGPOOL_LOCAL_H

#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

	void forward_avgpool_local(layer* l, network* net);
	void backward_avgpool_local(layer* l, network* net);
	void forward_avgpool_local_gpu(layer* l, network* net);
	void backward_avgpool_local_gpu(layer* l, network* net);

#ifdef __cplusplus
}
#endif
#endif