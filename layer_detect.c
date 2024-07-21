#include "layer_detect.h"


void forward_detect(layer* l, network* net) {
	float* l_truth = l->truth;
	size_t w = l->w;
	size_t h = l->h;
	size_t wh = w * h;
	size_t n = l->n;
	size_t n_classes = l->n_classes;
	size_t A = (NUM_ANCHOR_PARAMS + n_classes) * wh;  // array offset between anchors
	for (size_t b = 0; b < net->batch_size; b++) {
		float* b_pred = &l->in_layers[0]->output[b * n];
		for (size_t row = 0; row < w; row++) {
			size_t row_offset = row * h;
			for (size_t col = 0; col < h; col++) {
				size_t cell = row_offset + col;
				float* truth = &l->truth[cell];
				for (size_t a = 0; a < l->n_anchors; a++) {
					size_t anchor_offset = cell + a * A;
					float* p = &b_pred[anchor_offset];
					float anyobj = p[0];
					float px = p[wh];
					float py = p[wh * 2];
					float pw = p[wh * 3];
					float ph = p[wh * 4];
					for (size_t s = 0; s < n_classes; s++) {
						float cls = p[cell + (5 + s) * wh];
					}
				}
			}
		}
	}
	float* input = l->in_layers[0]->output;
	float objectness = 

}

#pragma warning(suppress:4100)
void backward_detect(layer* l, network* net) {
	l;

}