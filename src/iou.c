#include "iou.h"
#include <math.h>
#include <stdio.h>


#define EPS 0.000001F
#define square(x) ((x)*(x))
#define cube(x) ((x)*(x)*(x))

static const float PI = 3.14159265358979323846F;

static inline float clamp_x(float x, float thresh) {
	if (x > thresh) return thresh;
	if (x < -thresh) return -thresh;
	return x;
}


float get_iou(bbox pbox, bbox tbox) {
	// calculate intersection box
	float top = fmaxf(pbox.top, tbox.top);
	float bottom = fminf(pbox.bottom, tbox.bottom);
	if (bottom < top) return 0.0F;
	float left = fmaxf(pbox.left, tbox.left);
	float right = fminf(pbox.right, tbox.right);
	if (right < left) return 0.0F;
	float width = right - left;
	float height = bottom - top;
	float area = width * height;  // intersection area
	return area / (pbox.area + tbox.area - area);  // intersection area / union area
}

// Returns IOU, not 1 - IOU.
float get_grads_iou(bbox pbox, bbox tbox, float* dx, float* dy, float* dw, float* dh) {
	// note: derivatives are taken with respect to pbox
	// let pbox = box1, tbox = box2
	// dIOU/dx = dI/dx * (area1 + area2)/U^2
	// U_area = w1 * h1 + w2 * h2 - I_area
	// I_area = I_w * I_h
	// I_w = max(0, R_min - L_max)
	// I_h = max(0, B_min - T_max)
	// dI/dx = I_h * (dR_min/dx - dL_max/dx)
	float R = fminf(pbox.right, tbox.right);
	float L = fmaxf(pbox.left, tbox.left);
	float dR_dx = pbox.right < tbox.right ? 1.0F : 0.0F;
	float dL_dx = pbox.left > tbox.left ? 1.0F : 0.0F;
	float B = fminf(pbox.bottom, tbox.bottom);
	float T = fmaxf(pbox.top, tbox.top);
	float I_h = fmaxf(0.0F, B - T);
	float dI_dx = I_h * (dR_dx - dL_dx);
	float I_w = fmaxf(0.0F, R - L);
	float I = I_w * I_h;
	float U = pbox.area + tbox.area - I;
	float dIOU_dx = dI_dx * (pbox.area + tbox.area) / (U * U + EPS);

	// dIOU/dy = dI/dy * (area1 + area2)/U^2
	// dI/dy = I_w * (dB_min/dy - dT_max/dy)
	float dB_dy = pbox.bottom < tbox.bottom ? 1.0F : 0.0F;
	float dT_dy = pbox.top > tbox.top ? 1.0F : 0.0F;
	float dI_dy = I_w * (dB_dy - dT_dy);
	float dIOU_dy = dI_dy * (pbox.area + tbox.area) / (U * U + EPS);

	// dIOU/dw = (dI/dw * (area1 + area2) - I * h1) / U^2
	// dI/dw = I_h * (dR_min/dw - dL_max/dw)
	float dR_dw = pbox.right < tbox.right ? 0.5F : 0.0F;
	float dL_dw = pbox.left > tbox.left ? -0.5F : 0.0F;
	float dI_dw = I_h * (dR_dw - dL_dw);
	float dIOU_dw = (dI_dw * (pbox.area + tbox.area) - I * pbox.h) / (U * U + EPS);

	// dIOU/dh = (dI/dh * (area1 + area2) - I * w1) / U^2
	// dI/dh = I_w * (dB_min/dh - dT_max/dh)
	float dB_dh = pbox.bottom < tbox.bottom ? 0.5F : 0.0F;
	float dT_dh = pbox.top > tbox.top ? -0.5F : 0.0F;
	float dI_dh = I_w * (dB_dh - dT_dh);
	float dIOU_dh = (dI_dh * (pbox.area + tbox.area) - I * pbox.w) / (U * U + EPS);

	*dx = dIOU_dx;
	*dy = dIOU_dy;
	*dw = dIOU_dw;
	*dh = dIOU_dh;

	return I / (U + EPS);
}

float get_diou(bbox pbox, bbox tbox, float* iou) {
	// DIOU = IOU - C^2/E^2
	// C = distance between centers
	// E = diagonal of smallest enclosing box
	float C_sq = square(pbox.cx - tbox.cx) + square(pbox.cy - tbox.cy);
	float top = fminf(pbox.top, tbox.top);
	float bottom = fmaxf(pbox.bottom, tbox.bottom);
	float left = fminf(pbox.left, tbox.left);
	float right = fmaxf(pbox.right, tbox.right);
	float E_sq = square(right - left) + square(bottom - top);
	float _iou = get_iou(pbox, tbox);
	if (iou) *iou = _iou;
	return _iou - C_sq / (E_sq + EPS);
}

// Return DIOU, not 1 - DIOU.
float get_grads_diou(bbox pbox, bbox tbox, float* dx, float* dy, float* dw, float* dh, float* iou) {
	// dDIOU/d? = dIOU/d? - d(C^2/E^2)/d?
	// C = distance between centers
	// E = diagonal of smallest enclosing box
	float C2 = square(pbox.cx - tbox.cx) + square(pbox.cy - tbox.cy);
	float T = fmaxf(pbox.top, tbox.top);
	float B = fmaxf(pbox.bottom, tbox.bottom);
	float R = fmaxf(pbox.right, tbox.right);
	float L = fmaxf(pbox.left, tbox.left);
	float E2 = square(R - L) + square(B - T);
	float E4 = square(E2) + EPS;

	// Quotient Rule applied to DIOU exclusive term:
	// d(C^2/E^2)/d? = [d(C^2)/d? * E^2 - d(E^2)/d? * C^2] / E^4
	
	// d(C^2)/dx:
	float dC2_dx = 2.0F * (pbox.cx - tbox.cx);
	// d(E^2)/dx:
	float dR_dx = pbox.right > tbox.right ? 1.0F : 0.0F;
	float dL_dx = pbox.left < tbox.left ? 1.0F : 0.0F;
	float dE2_dx = 2.0F * (R - L) * (dR_dx - dL_dx);
	// d(C^2/E^2)/dx:
	float dC2E2_dx = (dC2_dx * E2 - dE2_dx * C2) / E4;

	// d(C^2)/dy:
	float dC2_dy = 2.0F * (pbox.cy - tbox.cy);
	// d(E^2)/dx:
	float dB_dy = pbox.bottom > tbox.bottom ? 1.0F : 0.0F;
	float dT_dy = pbox.top < tbox.top ? 1.0F : 0.0F;
	float dE2_dy = 2.0F * (B - T) * (dB_dy - dT_dy);
	// d(C^2/E^2)/dy:
	float dC2E2_dy = (dC2_dy * E2 - dE2_dy * C2) / E4;

	// d(C^2)/dw:
	// float dC2_dw = 0.0F;
	// d(E^2)/dw:
	float dR_dw = pbox.right > tbox.right ? 0.5F : 0.0F;
	float dL_dw = pbox.left < tbox.left ? -0.5F : 0.0F;
	float dE2_dw = 2.0F * (R - L) * (dR_dw - dL_dw);
	// d(C^2/E^2)/dw:
	float dC2E2_dw = -dE2_dw * C2 / E4;

	// d(C^2)/dh:
	// float dC2_dh = 0.0F;
	// d(E^2)/dh:
	float dB_dh = pbox.bottom > tbox.bottom ? 0.5F : 0.0F;
	float dT_dh = pbox.top < tbox.top ? -0.5F : 0.0F;
	float dE2_dh = 2 * (B - T) * (dB_dh - dT_dh);
	// d(C^2/E^2)/dh:
	float dC2E2_dh = -dE2_dh * C2 / E4;

	float dIOU_dx;
	float dIOU_dy;
	float dIOU_dw;
	float dIOU_dh;
	float _iou = get_grads_iou(pbox, tbox, &dIOU_dx, &dIOU_dy, &dIOU_dw, &dIOU_dh);

	*dx = dIOU_dx - dC2E2_dx;
	*dy = dIOU_dy - dC2E2_dy;
	*dw = dIOU_dw - dC2E2_dw;
	*dh = dIOU_dh - dC2E2_dh;

	if (iou) *iou = _iou;

	return _iou - (C2 / (E2 + EPS));
}

// Returns CIOU, not 1 - CIOU.
float get_ciou(bbox pbox, bbox tbox, float* diou, float* iou) {
	// CIOU = IOU - E^2/C^2 - alpha * v
	// aka CIOU = DIOU - alpha * v
	// let pbox = box1, tbox = box2
	// v = (4 / pi^2) * t
	// t = arctan(w2/h2) - arctan(w1/h1)
	// v = (4 / pi^2) * t^2
	float t = atanf(tbox.w / tbox.h) - atanf(pbox.w / pbox.h);
	float v = (4.0F / square(PI)) * square(t);
	// alpha = v / (1 - IOU + v) when IOU >= 0.5 else 0
	float _iou;
	float _diou = get_diou(pbox, tbox, &_iou);
	float alpha = (_iou < 0.5F) ? 0.0F : v / (1.0F - _iou + v + EPS);
	if (isnan(alpha) || isinf(alpha)) alpha = 0.0F;  // darknet does this so i'm assuming it's a good idea to include
	if (diou) *diou = _diou;
	if (iou) *iou = _iou;
	return _diou - alpha * v;
}

// Returns ciou loss as well as calculate gradients
float get_grads_ciou(bbox pbox, bbox tbox, float* dx, float* dy, float* dw, float* dh) {
	// dCIOU/d? = dIOU/d? - d(C^2/E^2)/d? - d(alpha * v)/d?
	// d(alpha * v)/d? = d(v^2 / (1 - IOU + v)/d?
	// let K = 1 - IOU
	// d(alpha * v)/d? = v * (dv/d?) * (2 * K + v) / (K + v)^2
	// v = (4 / pi^2) * t^2
	// t = atan(w2/h2) - atan(w1/h2)
	// dv/dw = (8 / pi^2) * (h1 / (h1^2 + w1^2)) * t
	// dv/dh = (8 / pi^2) * (-w1 / (h1^2 + w2^2)) * t
	// d(alpha * v)/dw = (v * dv/dw) * (2 * K + v) / (K + v)^2
	// d(alpha * v)/dh = (v * dv/dh) * (2 * K + v) / (K + v)^2
	float iou;
	float dDIOU_dx;
	float dDIOU_dy;
	float dDIOU_dw;
	float dDIOU_dh;
	float diou = get_grads_diou(pbox, tbox, &dDIOU_dx, &dDIOU_dy, &dDIOU_dw, &dDIOU_dh, &iou);

	float dav_dw = 0.0F;
	float dav_dh = 0.0F;
	float av = 0.0F;
	if (iou >= 0.5F) {
		float k = 1.0F - iou;
		float t = atanf(tbox.w / tbox.h) - atanf(pbox.w / pbox.h);
		float v = (4.0F / square(PI)) * square(t);

		float dv_dw = (8.0F / square(PI)) * (pbox.h / (square(pbox.h) + square(pbox.w))) * t;
		dav_dw = v * dv_dw * (2.0F * k + v) / square(k + v);

		float dv_dh = (8.0F / square(PI)) * (-pbox.w / (square(pbox.h) + square(pbox.w))) * t;
		dav_dh = v * dv_dh * (2.0F * k + v) / square(k + v);

		av = v * (k + v);
	}

	*dx = dDIOU_dx;
	*dy = dDIOU_dy;
	*dw = dDIOU_dw - dav_dw;
	*dh = dDIOU_dh - dav_dh;

	return diou - av;
}
