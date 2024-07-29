#include "iou.h"
#include <math.h>


#define V_CONST 4.0F / powf(acosf(-1.0F), 2.0F)  // used for calculating aspect ratio consistency for ciou
#define V2_CONST 8.0F / powf(acosf(-1.0F), 2.0F)

static inline float maxfloat(float x, float y) { return (x < y) ? y : x; }
static inline float minfloat(float x, float y) { return (x > y) ? y : x; }
static inline float absfloat(float x) { return (x < 0) ? -x : x; }
static inline float distance_between_points(float cx1, float cy1, float cx2, float cy2) {
	float dx = cx1 - cx2;
	float dy = cy1 - cy2;
	return sqrtf(dx * dx + dy * dy);
}


float get_iou(bbox box1, bbox box2) {
	// calculate intersection box
	float left = maxfloat(box1.left, box2.left);
	float top = maxfloat(box1.top, box2.top);
	float right = minfloat(box1.right, box2.right);
	float bottom = minfloat(box1.bottom, box2.bottom);
	float width = absfloat(right - left);
	float height = absfloat(bottom - top);
	float area = width * height;  // intersection area
	return area / (box1.area + box2.area - area);  // intersection area / union area
}

float get_diou(bbox box1, bbox box2) {
	float iou = get_iou(box1, box2);
	float delta = distance_between_points(box1.cx, box1.cy, box2.cx, box2.cy);
	// calculate diagonal of smallest enclosing box
	float left = minfloat(box1.left, box2.left);
	float top = minfloat(box1.top, box2.top);
	float right = maxfloat(box1.right, box2.right);
	float bottom = maxfloat(box1.bottom, box2.bottom);
	float diag = distance_between_points(left, top, right, bottom);
	return iou - (delta * delta) / (diag * diag);
}

float get_ciou(bbox box1, bbox box2) {
	float iou = get_iou(box1, box2);
	float delta = distance_between_points(box1.cx, box1.cy, box2.cx, box2.cy);
	// calculate diagonal of smallest enclosing box
	float left = minfloat(box1.left, box2.left);
	float top = minfloat(box1.top, box2.top);
	float right = maxfloat(box1.right, box2.right);
	float bottom = maxfloat(box1.bottom, box2.bottom);
	float diag = distance_between_points(left, top, right, bottom);
	// calculate aspect ratio consistency (v)
	float t = atanf(box2.w / box2.h) - atanf(box1.w / box1.h);
	float v = V_CONST * t * t;
	// calculate trade-off parameter for balancing the aspect ratio term
	float alpha = (iou < 0.5F) ? 0.0F : v / (1.0F - iou + v);
	return iou - (delta * delta) / (diag * diag) - alpha * v;
}

// Only calculates ciou loss, does not calculate any gradients
float loss_ciou(bbox box1, bbox box2) {
	float iou = get_iou(box1, box2);
	float delta = distance_between_points(box1.cx, box1.cy, box2.cx, box2.cy);
	// calculate diagonal of smallest enclosing box
	float left = minfloat(box1.left, box2.left);
	float top = minfloat(box1.top, box2.top);
	float right = maxfloat(box1.right, box2.right);
	float bottom = maxfloat(box1.bottom, box2.bottom);
	float diag = distance_between_points(left, top, right, bottom);
	// calculate aspect ratio consistency (v)
	float t = atanf(box2.w / box2.h) - atanf(box1.w / box1.h);
	float v = V_CONST * t * t;
	// calculate trade-off parameter for balancing the aspect ratio term
	float alpha = (iou < 0.5F) ? 0.0F : v / (1.0F - iou + v);
	return 1.0F - iou + (delta * delta) / (diag * diag) + alpha * v;
}

// Returns ciou loss as well as calculate gradients
float get_grads_ciou(bbox box1, bbox box2, float* dL_dx, float* dL_dy, float* dL_dw, float* dL_dh) {
	// intersection
	float I_left = maxfloat(box1.left, box2.left);
	float I_top = maxfloat(box1.top, box2.top);
	float I_right = minfloat(box1.right, box2.right);
	float I_bottom = minfloat(box1.bottom, box2.bottom);
	float I_w = absfloat(I_right - I_left);
	float I_h = absfloat(I_bottom - I_top);
	float I_area = I_w * I_h;
	// union
	float U_area = box1.area + box2.area - I_area;
	// iou
	float iou = I_area / U_area;

	// derivatives
	
	// dCIoU/dp = dIoU/dp - d(s^2/c^2)/dp - d(alpha * v)/dp
	// p = any given param (i.e. w, h, x, y)
	
	// intersection derivative
	// assume iou > 0 since the box should be filtered out before this point

	float dI_dL = box1.left < box2.left ? 0.0F : 1.0F;
	float dI_dR = box1.right < box2.right ? 1.0F : 0.0F;
	float dI_dT = box1.top < box2.top ? 0.0F : 1.0F;
	float dI_dB = box1.bottom < box2.bottom ? 1.0F : 0.0F;

	float dU_dL = box1.top - box1.bottom - dI_dL;
	float dU_dR = box1.bottom - box1.top - dI_dR;
	float dU_dT = box1.left - box1.right - dI_dT;
	float dU_dB = box1.right - box1.left - dI_dB;

	float dIoU_dL = (dI_dL * U_area - I_area * dU_dL) / (U_area * U_area);
	float dIoU_dR = (dI_dR * U_area - I_area * dU_dR) / (U_area * U_area);
	float dIoU_dT = (dI_dT * U_area - I_area * dU_dT) / (U_area * U_area);
	float dIoU_dB = (dI_dB * U_area - I_area * dU_dB) / (U_area * U_area);

	float dIoU_dw = dIoU_dR - dIoU_dL;
	float dIoU_dh = dIoU_dB - dIoU_dT;
	float dIoU_dx = dIoU_dL + dIoU_dR;
	float dIoU_dy = dIoU_dL + dIoU_dR;

	// smallest enclosing box that covers box1 and box2
	float C_left = minfloat(box1.left, box2.left);
	float C_top = minfloat(box1.top, box2.top);
	float C_right = maxfloat(box1.right, box2.right);
	float C_bottom = maxfloat(box1.bottom, box2.bottom);
	float C = distance_between_points(C_left, C_top, C_right, C_bottom);

	float dCL_dL = box1.left < box2.left ? 1.0F : 0.0F;
	float dC_dL = ((C_right - C_left) / C) * (-dCL_dL);
	float dCR_dR = box1.right > box2.right ? 1.0F : 0.0F;
	float dC_dR = ((C_right - C_left) / C) * dCR_dR;
	float dCT_dT = box1.top < box2.top ? 1.0F : 0.0F;
	float dC_dT = ((C_bottom - C_top) / C) * (-dCT_dT);
	float dCB_dB = box1.bottom > box2.bottom ? 1.0F : 0.0F;
	float dC_dB = ((C_bottom - C_top) / C) * dCB_dB;

	float dC_dw = dC_dR - dC_dL;
	float dC_dh = dC_dB - dC_dT;
	float dC_dx = dC_dL + dC_dR;
	float dC_dy = dC_dT + dC_dB;

	// distance between box1 and box2 centers
	float S = distance_between_points(box1.cx, box1.cy, box2.cx, box2.cy);
	float dS_dx = (box1.cx - box2.cx) / S;
	float dS_dy = (box1.cy - box2.cy) / S;
	float dS_dw = 0;
	float dS_dh = 0;

	float diou_term = (2.0F * S) / (C * C * C);
	float dDIoU_dx = diou_term * (C * dS_dx - S * dC_dx);
	float dDIoU_dy = diou_term * (C * dS_dy - S * dC_dy);
	float dDIoU_dw = diou_term * (C * dS_dw - S * dC_dw);
	float dDIoU_dh = diou_term * (C * dS_dh - S * dC_dh);

	// aspect ratio term
	float theta = atanf(box2.w / box2.h) - atanf(box1.w / box1.h);
	float V = V_CONST * theta * theta;
	float alpha = (iou < 0.5F) ? 0.0F : V / (1.0F - iou + V);

	float dV_dw = V2_CONST * theta * box1.h;
	float dV_dh = V2_CONST * theta * (-box2.w);
	// Note:
	// float dV_dx = 0;
	// float dV_dy = 0;

	*dL_dx = -dIoU_dx + dDIoU_dx;
	*dL_dy = -dIoU_dy + dDIoU_dy;
	*dL_dw = -dIoU_dw + dDIoU_dw + dV_dw * alpha;
	*dL_dh = -dIoU_dh + dDIoU_dh + dV_dh * alpha;

	return 1.0F - iou + S + V * alpha;
}