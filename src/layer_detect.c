#include "layer_detect.h"
#include <math.h>
#include "iou.h"
#include "utils.h"
#include "xallocs.h"
#include "image.h"


inline static void loss_cce_x(float x, float truth, float* error, float* grad);
void cull_predictions_and_do_nms(layer* l, network* net);
void draw_detections(bbox** dets, size_t n_dets, image* img, float thresh);
void pprint_detect_array(float* data, size_t rows, size_t cols, size_t n_classes, size_t n_anchors);


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
	float cell_size = (float)l_w / (float)net_w;
	size_t A = (NUM_ANCHOR_PARAMS + n_classes) * l_wh;
	float* p = l->in_layers[0]->output;
	float* errors = l->errors;
	zero_array(errors, l_n * batch_size);
	float* grads = l->grads;
	zero_array(grads, l_n * batch_size);
	float loss = l->loss;
	float obj_loss = l->obj_loss;
	float cls_loss = l->cls_loss;
	float iou_loss = l->iou_loss;
	loss = 0.0F;
	obj_loss = 0.0F;
	cls_loss = 0.0F;
	iou_loss = 0.0F;
	for (size_t s = 0; s < l_wh; s++) {
		det_cell cell = cells[s];
		for (size_t b = 0; b < batch_size; b++) {
			size_t bn = b * n_anchors;
			for (size_t a = 0; a < n_anchors; a++) {
				size_t bna = bn + a;
				// objectness
				size_t obj_index = s + b * l_n + a * A;
				float obj_truth = cell.obj[bna];
				loss_cce_x(p[obj_index], obj_truth, &errors[obj_index], &grads[obj_index]);
				obj_loss += errors[obj_index];
				// class
				for (size_t i = 0; i < n_classes; i++) {
					size_t cls_index = obj_index + (NUM_ANCHOR_PARAMS + i) * l_wh;
					float x = p[cls_index];
					float t = (cell.cls[bna] == (int)i) ? 1.0F : 0.0F;
					loss_cce_x(x, t, &errors[cls_index], &grads[cls_index]);
					cls_loss += errors[cls_index];
				}
				if (obj_truth < 1.0F) continue;
				// iou
				float cx = p[obj_index + l_wh] * cell_size + cell.left;  // predicted value for cx, cy is % of cell size
				float cy = p[obj_index + l_wh * 2] * cell_size + cell.top;
				float w = p[obj_index + l_wh * 3];  // predicted value for w, h is % of img size
				float h = p[obj_index + l_wh * 4];
				bbox pbox = { 0 };
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
				float iouloss = get_grads_ciou(pbox, cell.tboxes[bna], dL_dx, dL_dy, dL_dw, dL_dh);
				//printf("Cell[%zu] Anchor[%zu] iou loss: %f\n", s, a, iouloss);
				iou_loss += iouloss;
			}
		}
	}
	l->loss = obj_loss + cls_loss + iou_loss;
	l->obj_loss = obj_loss;
	l->cls_loss = cls_loss;
	l->iou_loss = iou_loss;
	printf("detect loss: %f\nobj loss: %f\nclass loss: %f\niou loss: %f\n", l->loss, obj_loss, cls_loss, iou_loss);
	//pprint_detect_array(l->grads, l->h, l->w, n_classes, n_anchors);
	cull_predictions_and_do_nms(l, net);  // for debugging
}

void backward_detect(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* l_grads = l->grads;
	size_t l_out_n = l->out_n;
	layer** inls = l->in_layers;
	size_t b;
#pragma omp parallel for firstprivate(l_grads, l_out_n, inls)
	for (b = 0; b < batch_size; b++) {
		float* b_grads = &l_grads[b * l_out_n];
		for (size_t i = 0; i < l->in_ids.n; i++) {
			layer* inl = inls[i];
			size_t inl_out_n = inl->out_n;
			float* inl_grads = &inl->grads[b * inl_out_n];
			for (size_t g = 0; g < inl_out_n; g++) {
				inl_grads[g] = b_grads[g];
			}
			b_grads += inl_out_n;
		}
	}
}

void cull_predictions_and_do_nms(layer* l, network* net) {
	size_t net_w = net->w;
	/*float obj_thresh = l->nms_obj_thresh;
	float cls_thresh = l->nms_cls_thresh;*/
	float iou_thresh = l->nms_iou_thresh;
	float obj_thresh = 0.0F;
	float cls_thresh = 0.0F;
	det_cell* cells = l->cells;
	size_t batch_size = net->batch_size;
	size_t l_w = l->w;
	size_t l_h = l->h;
	size_t l_wh = l_w * l_h;
	size_t l_n = l->n;
	size_t n_classes = l->n_classes;
	size_t n_anchors = l->n_anchors;
	float cell_size = (float)l_w / (float)net_w;
	size_t A = (NUM_ANCHOR_PARAMS + n_classes) * l_wh;
	float* p = l->in_layers[0]->output;
	size_t ndets = 0;  // debugging
	for (size_t b = 0; b < batch_size; b++) {
		bbox* dets = l->detections;
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
				float left = cx - w / 2.0F;
				if (left >= 1.0F) continue;
				float right = left + w;
				if (right <= 0.0F) continue;
				float top = cy - w / 2.0F;
				if (top >= 1.0F) continue;
				float bottom = top + h;
				if (bottom <= 0.0F) continue;
				dets->prob = best_cls_score;
				dets->lbl = best_cls;
				if (left < 0.0F) left = 0.0F;
				if (right > 1.0F) right = 1.0F;
				if (top < 0.0F) top = 0.0F;
				if (bottom > 1.0F) bottom = 1.0F;
				dets->cx = (left + right) / 2.0F;
				dets->cy = (top + bottom) / 2.0F;
				w = right - left;
				h = bottom - top;
				dets->w = w;
				dets->h = h;
				dets->area = w * h;
				dets->left = left;
				dets->right = right;
				dets->top = top;
				dets->bottom = bottom;
				*sorted = dets;
				sorted++;
				dets++;
				n_dets++;
			}
		}
		if (n_dets == 0) continue;
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
		if (b == 0) ndets = n_dets;
	}
	// draw boxes on input image
	det_sample** samples = net->data.detr.current_batch;
	image img = { 0 };
	float buffer[3072] = { 0 };
	img.w = 32;
	img.h = 32;
	img.c = 3;
	img.data = buffer;
	load_image_to_buffer(samples[0]->imgpath, &img);
	net->draw_thresh = 0.5F;  // TODO: Make a cfg parameter
	draw_detections(l->sorted, ndets, &img, net->draw_thresh);
	////write_image(img, "D:\\TonyDev\\Nardenet\\data\\detector\\test.png");
	show_image(&img);
	//xfree(img);
}

void draw_detections(bbox** dets, size_t n_dets, image* img, float thresh) {
	float* data = img->data;
	size_t img_w = img->w;
	size_t img_h = img->h;
	float red = 255.0F;
	float green = 0.0F;
	float blue = 0.0F;
	size_t red_offset = 0 * img_w * img_h;
	size_t green_offset = 1 * img_w * img_h;
	size_t blue_offset = 2 * img_w * img_h;
	for (size_t i = 0; i < n_dets; i++) {
		if (dets[i]->prob < thresh) continue;
		bbox* det = dets[i];
		size_t box_left = det->left * img_w;
		size_t box_top = det->top * img_h;
		size_t box_right = (det->right * img_w) - 1;
		size_t box_bottom = (det->bottom * img_h) - 1;
		// draw horizontal lines
		size_t row_offset_top =  box_top * img_w;
		size_t row_offset_bottom = box_bottom * img_w;
		size_t col = box_left;
		while (col < box_right) {
			size_t offset_top = row_offset_top + col;
			size_t offset_bottom = row_offset_bottom + col;
			data[red_offset + offset_top] = red;
			data[red_offset + offset_bottom] = red;
			data[green_offset + offset_top] = green;
			data[green_offset + offset_bottom] = green;
			data[blue_offset + offset_top] = blue;
			data[blue_offset + offset_bottom] = blue;
			if (blue_offset + offset_bottom >= 3072) printf("offset: %zu\n", blue_offset + offset_bottom);
			col++;
		}
		// draw vertical lines
		size_t row = box_top;
		while (row < box_bottom) {
			size_t row_offset = row * img_w;
			size_t offset_left = row_offset + box_left;
			size_t offset_right = row_offset + box_right;
			data[red_offset + offset_left] = red;
			data[red_offset + offset_right] = red;
			data[green_offset + offset_left] = green;
			data[green_offset + offset_right] = green;
			data[blue_offset + offset_left] = blue;
			data[blue_offset + offset_right] = blue;
			row++;
		}
	}
}

void pprint_detect_array(float* data, size_t rows, size_t cols, size_t n_classes, size_t n_anchors) {
	size_t S = rows * cols;
	size_t A = (NUM_ANCHOR_PARAMS + n_classes) * S;
	for (size_t row = 0; row < rows; row++) {
		for (size_t col = 0; col < cols; col++) {
			for (size_t a = 0; a < n_anchors; a++) {
				size_t s = row * rows + col;
				size_t obji = s + A * a;
				size_t cxi = obji + S;
				size_t cyi = cxi + S;
				size_t wi = cyi + S;
				size_t hi = wi + S;
				printf("\nobj: %0.3f, cx: %0.3f, cy: %0.3f, w: %0.3f, h: %0.3f\n", data[obji], data[cxi], data[cyi], data[wi], data[hi]);
				printf("Classes: ");
				for (size_t n = 0; n < n_classes; n++) {
					size_t clsi = hi + S + (S * n);
					printf("[%zu]:%0.3f ", n, data[clsi]);
				}
				printf("\n");
			}
		}
	}
	printf("Press enter to continue.\n");
	(void)getchar();
}

inline static void loss_cce_x(float x, float truth, float* error, float* grad) {
	*grad = x - truth;
	*error = -truth * logf(x) - (1.0F - truth) * logf(1.0F - x);
}