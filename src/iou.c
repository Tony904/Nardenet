#include "iou.h"


static inline float maxfloat(float x, float y) { return (x < y) ? y : x; }
static inline float minfloat(float x, float y) { return (x > y) ? y : x; }
static inline float absfloat(float x) { return (x < 0) ? -x : x; }


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