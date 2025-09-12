#ifndef LAYER_CONV_H
#define LAYER_CONV_H


#include "network.h"


#ifdef __cplusplus
extern "C" {
#endif

	void forward_fc(layer* l, network* net);
	void backward_fc(layer* l, network* net);
	void forward_fc_gpu(layer* l, network* net);
	void backward_fc_gpu(layer* l, network* net);

#ifdef __cplusplus
}
#endif
#endif