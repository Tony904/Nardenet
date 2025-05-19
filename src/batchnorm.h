#ifndef BATCHNORM_H
#define BATCHNORM_H

#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

	void forward_batchnorm(layer* l, size_t batch_size);
	void backward_batchnorm(layer* l, size_t batch_size);

#ifdef __cplusplus
}
#endif
#endif