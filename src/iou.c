#include "iou.h"
#include <math.h>


#define V_CONST 0.40528473F  // (4/pi^2) used for calculating aspect ratio consistency for ciou


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

void get_gradients_ciou(bbox box1, bbox box2) {
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
	// intersection derivative
	float dIw_dx = (box1.left <= box2.left && box2.left < box1.right) ?  1.0F :
				  (box2.left <= box1.left && box1.left < box2.right) ? -1.0F : 0.0F;
	float dI_dx = dIw_dx * I_h;
	float dIh_dy = (box1.top <= box2.top && box2.top < box1.bottom) ?  1.0F :
				  (box2.top <= box1.top && box1.top < box2.bottom) ? -1.0F : 0.0F;
	float dI_dy = dIh_dy * I_w;

	float dIw_dw = 
}