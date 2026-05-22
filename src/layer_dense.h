#ifndef LAYER_DENSE_H
#define LAYER_DENSE_H


#include "network.h"


#ifdef __cplusplus
extern "C" {
#endif

	void forward_dense(layer* l, network* net);
	void backward_dense(layer* l, network* net);
	void forward_dense_gpu(layer* l, network* net);
	void backward_dense_gpu(layer* l, network* net);

	void forward_dense_cpu_gpu_compare(layer* l, network* net);
	void backward_dense_cpu_gpu_compare(layer* l, network* net);

#ifdef __cplusplus
}
#endif
#endif