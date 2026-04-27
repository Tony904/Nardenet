#ifndef IOU_H
#define IOU_H

#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

	float get_iou(bbox pbox, bbox tbox);
	float get_grads_iou(bbox pbox, bbox tbox, float* dx, float* dy, float* dw, float* dh);
	float get_diou(bbox pbox, bbox tbox, float* iou);
	float get_grads_diou(bbox pbox, bbox tbox, float* dx, float* dy, float* dw, float* dh, float* iou);
	float get_ciou(bbox pbox, bbox tbox, float* diou, float* iou);
	float get_grads_ciou(bbox pbox, bbox tbox, float* dx, float* dy, float* dw, float* dh);

#ifdef __cplusplus
}
#endif
#endif