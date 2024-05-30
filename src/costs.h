#ifndef COSTS_H
#define COSTS_H

#include <stdio.h>
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

	void cost_mse(layer* l);
	void cost_softmax_cce(layer* l);
	void cost_sigmoid_cce(layer* l);
	void cost_bce(layer* l);

#ifdef __cplusplus
}
#endif
#endif