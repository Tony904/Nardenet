#ifndef LAYER_RESIDUAL_H
#define LAYER_RESIDUAL_H

#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

	void forward_residual(layer* l, network* net);
	void backward_residual(layer* l, network* net);
	void forward_residual_gpu(layer* l, network* net);
	void backward_residual_gpu(layer* l, network* net);

#ifdef __cplusplus
}
#endif
#endif