#ifndef LAYER_UPSAMPLE_H
#define LAYER_UPSAMPLE_H

#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

	void forward_upsample(layer* l, network* net);
	void backward_upsample(layer* l, network* net);
	void forward_upsample_gpu(layer* l, network* net);
	void backward_upsample_gpu(layer* l, network* net);
	/*void test_forward_upsample(void);
	void test_backward_upsample(void);*/

#ifdef __cplusplus
}
#endif
#endif