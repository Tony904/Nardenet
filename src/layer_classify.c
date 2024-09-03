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
	l->get_loss(l, net);
}
