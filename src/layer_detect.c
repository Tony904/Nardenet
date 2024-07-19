#include "layer_detect.h"


void forward_detect(layer* l, network* net) {
	// calculate any_object loss, class loss, iou loss
	// note: types of iou loss: standard iou, giou, diou, ciou


	// convert sigmoid output to useable bounding box coords for drawing to image

}

#pragma warning(suppress:4100)
void backward_detect(layer* l, network* net) {
	l;

}

void get_best_anchor(bbox box, layer* l, float cell_left, float cell_top, float cell_size) {
	float iou = 0.0F;
	cx -= cell_left;
	cy -= cell_top;
	for (size_t a = 0; a < l->n_anchors; a++) {
		float* anchor = &l->anchors[a * 2];
		float anchor_w = anchor[0];
		float anchor_h = anchor[1];
		float anchor_left = cell_left + 0.5F * (cell_size - anchor_w);
		float anchor_top = cell_top + 0.5F * (cell_size - anchor_h);
		float anchor_right = cell_left + 0.5F * (cell_size + anchor_w);
		float anchor_bottom = cell_left + 0.5F * (cell_size + anchor_h);

	}
}