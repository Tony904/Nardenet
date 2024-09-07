#ifndef LAYER_RESIDUAL_H
#define LAYER_RESIDUAL_H

#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

	void forward_residual(layer* l, network* net);
	void backward_residual(layer* l, network* net);
	/*void test_forward_residual(void);
	void test_backward_residual(void);*/

#ifdef __cplusplus
}
#endif
#endif