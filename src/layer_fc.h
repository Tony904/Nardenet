#ifndef LAYER_FC_H
#define LAYER_FC_H


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