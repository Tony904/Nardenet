#ifndef LOSS_H
#define LOSS_H

#include <stdio.h>
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

	void loss_mse(layer* l, size_t batch_size);
	void loss_softmax_cce(layer* l, size_t batch_size);
	void loss_sigmoid_cce(layer* l, size_t batch_size);
	void loss_bce(layer* l, size_t batch_size);
	void loss_l1(network* net);
	void loss_l2(network* net);

#ifdef __cplusplus
}
#endif
#endif