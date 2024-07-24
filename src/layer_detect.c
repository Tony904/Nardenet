#include "layer_detect.h"
#include <math.h>
#include "iou.h"


inline static void loss_cce_x(float x, float truth, float* error, float* grad);


void forward_detect(layer* l, network* net) {
	det_cell* cells = l->cells;
	size_t batch_size = net->batch_size;
	size_t l_w = l->w;
	size_t l_h = l->h;
	size_t l_wh = l_w * l_h;
	size_t l_n = l->n;
	size_t n_classes = l->n_classes;
	size_t n_anchors = l->n_anchors;
	size_t net_w = net->w;
	det_sample* samples = net->data.detr.samples;
	float cell_size = (float)l_w / (float)net_w;
	size_t A = (NUM_ANCHOR_PARAMS + n_classes) * l_wh;
	float* p = l->in_layers[0]->output;
	for (size_t s = 0; s < l_wh; s++) {
		det_cell cell = cells[s];
		for (size_t b = 0; b < batch_size; b++) {
			det_sample sample = samples[b];
			size_t bn = b * n_anchors;
			for (size_t a = 0; a < n_anchors; a++) {
				size_t bna = bn + a;
				// objectness
				size_t index = s + b * l_n + a * A;
				float* error = &l->errors[index];
				float* grad = &l->grads[index];
				loss_cce_x(p[index], (float)cell.obj[bna], error, grad);
				l->loss += *error;
				// class
				for (size_t i = 0; i < n_classes; i++) {
					size_t index2 = index + (NUM_ANCHOR_PARAMS + i) * l_wh;
					float x = p[index2];
					float t = (cell.cls[bna] == (int)i) ? 1.0F : 0.0F;
					loss_cce_x(x, t, &error[index2], &grad[index2]);
					l->loss += error[index2];
				}
				// iou
				bbox pbox = { 0 };
				float cx = p[index + l_wh] * cell_size + cell.left;  // note: predicted value is % of cell size
				float cy = p[index + l_wh * 2] * cell_size + cell.top;
				float w = p[index + l_wh * 3];
				float h = p[index + l_wh * 4];
				pbox.cx = cx;
				pbox.cy = cy;
				pbox.w = w;
				pbox.h = h;
				pbox.area = w * h;
				pbox.left = cell.left + cx * cell_size - w;
				pbox.right = pbox.left + w;
				pbox.top = cell.top + cx * cell_size - h;
				pbox.bottom = cell.top + h;
				
				float ciou = loss_ciou(pbox, cell.tboxes[bna]);

			}
		}
	}
}

#pragma warning(suppress:4100)
void backward_detect(layer* l, network* net) {
	l;

}

inline static void loss_cce_x(float x, float truth, float* error, float* grad) {
	*grad = x - truth;
	*error = -truth * logf(x) - (1.0F - truth) * logf(1.0F - x);
}