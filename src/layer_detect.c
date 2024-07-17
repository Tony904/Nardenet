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