#ifndef LOSS_H
#define LOSS_H

#include <stdio.h>
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

	void loss_mae(layer* l, network* net);
	void loss_mse(layer* l, network* net);
	void loss_softmax_cce(layer* l, network* net);
	void loss_sigmoid_cce(layer* l, network* net);
	void loss_l1(network* net);
	void loss_l2(network* net);

#ifdef __cplusplus
}
#endif
#endif