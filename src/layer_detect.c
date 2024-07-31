#include "layer_detect.h"
#include <math.h>
#include "iou.h"
#include "utils.h"
#include "xallocs.h"
#include "image.h"


inline static void loss_cce_x(float x, float truth, float* error, float* grad);
void cull_predictions_and_do_nms(layer* l, network* net);


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
	float* errors = &l->errors;
	zero_array(errors, l_n * batch_size);
	float* grads = &l->grads;
	zero_array(grads, l_n * batch_size);
	float loss = l->loss;
	loss = 0.0F;
	for (size_t s = 0; s < l_wh; s++) {
		det_cell cell = cells[s];
		for (size_t b = 0; b < batch_size; b++) {
			det_sample sample = samples[b];
			size_t bn = b * n_anchors;
			for (size_t a = 0; a < n_anchors; a++) {
				size_t bna = bn + a;
				// objectness
				size_t obj_index = s + b * l_n + a * A;
				loss_cce_x(p[obj_index], (float)cell.obj[bna], &errors[obj_index], &grads[obj_index]);
				l->loss += errors[obj_index];
				// class
				for (size_t i = 0; i < n_classes; i++) {
					size_t cls_index = obj_index + (NUM_ANCHOR_PARAMS + i) * l_wh;
					float x = p[cls_index];
					float t = (cell.cls[bna] == (int)i) ? 1.0F : 0.0F;
					loss_cce_x(x, t, &errors[cls_index], &grads[cls_index]);
					l->loss += errors[cls_index];
				}
				// iou
				bbox pbox = { 0 };
				float cx = p[obj_index + l_wh] * cell_size + cell.left;  // predicted value for cx, cy is % of cell size
				float cy = p[obj_index + l_wh * 2] * cell_size + cell.top;
				float w = p[obj_index + l_wh * 3];  // predicted value for w, h is % of img size
				float h = p[obj_index + l_wh * 4];
				pbox.cx = cx;
				pbox.cy = cy;
				pbox.w = w;
				pbox.h = h;
				pbox.area = w * h;
				pbox.left = cx - (w / 2.0F);
				pbox.right = pbox.left + w;
				pbox.top = cy - (h / 2.0F);
				pbox.bottom = cell.top + h;
				float* dL_dx = &grads[obj_index + l_wh];
				float* dL_dy = &grads[obj_index + l_wh * 2];
				float* dL_dw = &grads[obj_index + l_wh * 3];
				float* dL_dh = &grads[obj_index + l_wh * 4];
				loss += get_grads_ciou(pbox, cell.tboxes[bna], dL_dx, dL_dy, dL_dw, dL_dh);
			}
		}
	}
	l->loss = loss;
	draw_detections(l, net);  // for debugging
}

#pragma warning(suppress:4100)
void backward_detect(layer* l, network* net) {
	l;
}

void cull_predictions_and_do_nms(layer* l, network* net) {
	size_t net_w = net->w;
	size_t net_h = net->h;
	size_t net_c = net->c;
	size_t net_input_size = net_w * net_h * net_c;
	float obj_thresh = l->nms_obj_thresh;
	float cls_thresh = l->nms_cls_thresh;
	float iou_thresh = l->nms_iou_thresh;
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
	bbox* dets = l->detections;
	for (size_t b = 0; b < batch_size; b++) {
		bbox** sorted = l->sorted;
		size_t n_dets = 0;
		// cull predictions below thresholds
		for (size_t s = 0; s < l_wh; s++) {
			det_cell cell = cells[s];
			for (size_t a = 0; a < n_anchors; a++) {
				size_t obj_index = s + b * l_n + a * A;
				if (p[obj_index] < obj_thresh) continue;  // objectness thresh
				int best_cls = -1;
				float best_cls_score = 0.0F;
				for (size_t i = 0; i < n_classes; i++) {
					size_t cls_index = obj_index + (NUM_ANCHOR_PARAMS + i) * l_wh;
					if (p[cls_index] > cls_thresh) {  // class confidence thresh
						if (p[cls_index] > best_cls_score) {
							best_cls_score = p[cls_index];
							best_cls = (int)i;
						}
					}
				}
				if (best_cls < 0) continue;
				float cx = p[obj_index + l_wh] * cell_size + cell.left;  // predicted value for cx, cy is % of cell size
				float cy = p[obj_index + l_wh * 2] * cell_size + cell.top;
				float w = p[obj_index + l_wh * 3];  // predicted value for w, h is % of img size
				float h = p[obj_index + l_wh * 4];
				dets->cx = cx;
				dets->cy = cy;
				dets->w = w;
				dets->h = h;
				dets->prob = best_cls_score;
				dets->lbl = best_cls;
				dets->left = cx - w / 2.0F;
				dets->right = cx + w / 2.0F;
				dets->top = cy - w / 2.0F;
				dets->bottom = cy + w / 2.0F;
				dets->area = w * h;
				*sorted = dets;
				sorted++;
				dets++;
				n_dets++;
			}
		}
		// sort remaining detections
		dets = l->detections;
		sorted = l->sorted;
		size_t i = 0;
		while (i < n_dets - 1) {
			if (sorted[i]->prob < sorted[i + 1]->prob) {
				bbox* tmp = sorted[i];
				sorted[i] = sorted[i + 1];
				sorted[i + 1] = tmp;
				i = i ? i - 1 : i + 1;
			}
			else i++;
		}
		// nms
		for (i = 0; i < n_dets; i++) {
			bbox* test = sorted[i];
			for (size_t j = i + 1; j < n_dets; j++) {
				if (test->lbl == sorted[j]->lbl) {
					float iou = get_diou(*test, *sorted[j]);
					if (iou < iou_thresh) {
						sorted[j]->prob = 0.0F;
						sorted[j]->lbl = -1;
					}
				}
			}
		}
		// draw boxes on input image
		// TODO: Need to store copy of input images for drawing. Can't just use data copied to input layer.
		draw_detections(sorted, &net->input->output[b * net_input_size], net_w, net_h, net_c);
	}
}

void draw_detections(bbox** sorted, float* img_data, size_t w, size_t h, size_t c) {

}

inline static void loss_cce_x(float x, float truth, float* error, float* grad) {
	*grad = x - truth;
	*error = -truth * logf(x) - (1.0F - truth) * logf(1.0F - x);
}