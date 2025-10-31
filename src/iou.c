#include "iou.h"
#include <math.h>
#include <stdio.h>


#define V_CONST 4.0F / powf(acosf(-1.0F), 2.0F)  // used for calculating aspect ratio consistency for ciou
#define V2_CONST 8.0F / powf(acosf(-1.0F), 2.0F)


static inline float clamp_x(float x, float thresh);


static inline float distance_between_points(float cx1, float cy1, float cx2, float cy2) {
	float dx = cx1 - cx2;
	float dy = cy1 - cy2;
	return sqrtf(dx * dx + dy * dy);
}


float get_iou(bbox box1, bbox box2) {
	// calculate intersection box
	float left = fmaxf(box1.left, box2.left);
	float top = fmaxf(box1.top, box2.top);
	float right = fminf(box1.right, box2.right);
	float bottom = fminf(box1.bottom, box2.bottom);
	float width = fabsf(right - left);
	float height = fabsf(bottom - top);
	float area = width * height;  // intersection area
	return area / (box1.area + box2.area - area);  // intersection area / union area
}

float get_diou(bbox box1, bbox box2) {
	float iou = get_iou(box1, box2);
	float delta = distance_between_points(box1.cx, box1.cy, box2.cx, box2.cy);
	// calculate diagonal of smallest enclosing box
	float left = fminf(box1.left, box2.left);
	float top = fminf(box1.top, box2.top);
	float right = fmaxf(box1.right, box2.right);
	float bottom = fmaxf(box1.bottom, box2.bottom);
	float diag = distance_between_points(left, top, right, bottom);
	return iou - (delta * delta) / (diag * diag);
}

float get_ciou(bbox box1, bbox box2) {
	float iou = get_iou(box1, box2);
	float delta = distance_between_points(box1.cx, box1.cy, box2.cx, box2.cy);
	// calculate diagonal of smallest enclosing box
	float left = fminf(box1.left, box2.left);
	float top = fminf(box1.top, box2.top);
	float right = fmaxf(box1.right, box2.right);
	float bottom = fmaxf(box1.bottom, box2.bottom);
	float diag = distance_between_points(left, top, right, bottom);
	// calculate aspect ratio consistency (v)
	float t = atanf(box2.w / box2.h) - atanf(box1.w / box1.h);
	float v = V_CONST * t * t;
	// calculate trade-off parameter for balancing the aspect ratio term
	float alpha = (iou < 0.5F) ? 0.0F : v / (1.0F - iou + v);
	if (isnan(alpha) || isinf(alpha)) alpha = 0.0F;
	return iou - (delta * delta) / (diag * diag) - alpha * v;
}

// Returns ciou loss as well as calculate gradients
float get_grads_ciou(bbox box1, bbox box2, float* dL_dx, float* dL_dy, float* dL_dw, float* dL_dh, float max_box_grad) {
	float ep = 0.000001F;
	// intersection
	float I_left = fmaxf(box1.left, box2.left);
	float I_top = fmaxf(box1.top, box2.top);
	float I_right = fminf(box1.right, box2.right);
	float I_bottom = fminf(box1.bottom, box2.bottom);
	float I_w = fabsf(I_right - I_left);
	float I_h = fabsf(I_bottom - I_top);
	float I_area = I_w * I_h;
	// union
	float U_area = box1.area + box2.area - I_area;
	// iou
	float iou = I_area / (U_area + ep);

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

	float dIoU_dL = (dI_dL * U_area - I_area * dU_dL) / (U_area * U_area + ep);
	float dIoU_dR = (dI_dR * U_area - I_area * dU_dR) / (U_area * U_area + ep);
	float dIoU_dT = (dI_dT * U_area - I_area * dU_dT) / (U_area * U_area + ep);
	float dIoU_dB = (dI_dB * U_area - I_area * dU_dB) / (U_area * U_area + ep);

	float dIoU_dw = dIoU_dR - dIoU_dL;
	float dIoU_dh = dIoU_dB - dIoU_dT;
	float dIoU_dx = dIoU_dL + dIoU_dR;
	float dIoU_dy = dIoU_dL + dIoU_dR;

	// smallest enclosing box that covers box1 and box2
	float C_left = fminf(box1.left, box2.left);
	float C_top = fminf(box1.top, box2.top);
	float C_right = fmaxf(box1.right, box2.right);
	float C_bottom = fmaxf(box1.bottom, box2.bottom);
	float C = distance_between_points(C_left, C_top, C_right, C_bottom);

	float dCL_dL = box1.left < box2.left ? 1.0F : 0.0F;
	float dC_dL = ((C_right - C_left) / (C + ep)) * (-dCL_dL);
	float dCR_dR = box1.right > box2.right ? 1.0F : 0.0F;
	float dC_dR = ((C_right - C_left) / (C + ep)) * dCR_dR;
	float dCT_dT = box1.top < box2.top ? 1.0F : 0.0F;
	float dC_dT = ((C_bottom - C_top) / (C + ep)) * (-dCT_dT);
	float dCB_dB = box1.bottom > box2.bottom ? 1.0F : 0.0F;
	float dC_dB = ((C_bottom - C_top) / (C + ep)) * dCB_dB;

	float dC_dw = dC_dR - dC_dL;
	float dC_dh = dC_dB - dC_dT;
	float dC_dx = dC_dL + dC_dR;
	float dC_dy = dC_dT + dC_dB;

	// distance between box1 and box2 centers
	float S = distance_between_points(box1.cx, box1.cy, box2.cx, box2.cy);
	float dS_dx = (box1.cx - box2.cx) / (S + ep);
	float dS_dy = (box1.cy - box2.cy) / (S + ep);
	float dS_dw = 0;
	float dS_dh = 0;

	float diou_term = (2.0F * S) / (C * C * C + ep);
	float dDIoU_dx = diou_term * (C * dS_dx - S * dC_dx);
	float dDIoU_dy = diou_term * (C * dS_dy - S * dC_dy);
	float dDIoU_dw = diou_term * (C * dS_dw - S * dC_dw);
	float dDIoU_dh = diou_term * (C * dS_dh - S * dC_dh);

	// aspect ratio term
	float theta = atanf(box2.w / box2.h) - atanf(box1.w / box1.h);
	float V = V_CONST * theta * theta;
	float alpha = (iou < 0.5F) ? 0.0F : V / (1.0F - iou + V);
	if (isnan(alpha) || isinf(alpha)) alpha = 0.0F;

	float dV_dw = V2_CONST * theta * box1.h;
	float dV_dh = V2_CONST * theta * (-box2.w);
	// Note:
	// float dV_dx = 0;
	// float dV_dy = 0;

	*dL_dw = clamp_x(*dL_dw, max_box_grad);
	*dL_dh = clamp_x(*dL_dh, max_box_grad);
	*dL_dx = clamp_x(*dL_dx, max_box_grad);
	*dL_dy = clamp_x(*dL_dy, max_box_grad);

	int accumulate = 1;  // will make cfg parameter later. probably...
	if (!accumulate) {
		*dL_dx = 0.0F;
		*dL_dy = 0.0F;
		*dL_dw = 0.0F;
		*dL_dh = 0.0F;
	}

	*dL_dx += clamp_x(-dIoU_dx + dDIoU_dx, max_box_grad);
	*dL_dy += clamp_x(-dIoU_dy + dDIoU_dy, max_box_grad);
	*dL_dw += clamp_x(-dIoU_dw + dDIoU_dw + dV_dw * alpha, max_box_grad);
	*dL_dh += clamp_x(-dIoU_dh + dDIoU_dh + dV_dh * alpha, max_box_grad);
	//printf("diou_dw: %f dDiou_dw: %f dv_dw: %f alpha: %f\n", dIoU_dw, dDIoU_dw, dV_dw, alpha);
	//printf("diou_dh: %f dDiou_dh: %f dv_dh: %f alpha: %f\n", dIoU_dh, dDIoU_dh, dV_dh, alpha);

	return 1.0F - iou + S + V * alpha;
}

inline static float clamp_x(float x, float thresh) {
	if (x > thresh) return thresh;
	if (x < -thresh) return -thresh;
	return x;
}