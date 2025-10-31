#ifndef IOU_H
#define IOU_H

#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

	float get_iou(bbox box1, bbox box2);
	float get_diou(bbox box1, bbox box2);
	float get_ciou(bbox box1, bbox box2);
	float get_grads_ciou(bbox box1, bbox box2, float* dL_dx, float* dL_dy, float* dL_dw, float* dL_dh, float max_box_grad);

#ifdef __cplusplus
}
#endif
#endif