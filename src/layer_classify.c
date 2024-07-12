#include "layer_classify.h"
#include <omp.h>
#include <assert.h>
#include <stdio.h>
#include "xallocs.h"
#include "im2col.h"
#include "gemm.h"
#include "network.h"
#include "activations.h"
#include "loss.h"
#include "utils.h"
#include "derivatives.h"
#include "layer_conv.h"


void forward_classify(layer* l, network* net) {
	forward_conv(l, net);
	l->get_loss(l, net);
}

#pragma warning(suppress:4100)  // unreferenced formal parameter: 'net'
void backward_classify(layer* l, network* net) {
	backward_conv(l, net);
}

void update_classify(layer* l, network* net) {
	update_conv(l, net);
}