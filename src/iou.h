#ifndef IOU_H
#define IOU_H

#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

	float get_iou(bbox box1, bbox box2);
	float get_ciou(bbox box1, bbox box2);
	float loss_ciou(bbox box1, bbox box2);

#ifdef __cplusplus
}
#endif
#endif