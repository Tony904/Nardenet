#include "layer_conv.h"
#include "xallocs.h"
#include "im2col.h"


void forward_layer_first(layer* l, network* net) {
	image* img = net->input;
	//float* weights = l->weights.a;
	//float* output = l->output;

	float* xmat = (float*)xcalloc(l->out_n * l->ksize * l->ksize, sizeof(float));
	im2col_cpu(img->data, (int)l->c, (int)l->h, (int)l->w, (int)l->ksize, (int)l->pad, (int)l->stride, xmat);


}


//void forward_layer_conv(layer* l, network* net) {
//	
//}
//
//
//
//void backward_layer_conv(layer* l) {
//	
//}