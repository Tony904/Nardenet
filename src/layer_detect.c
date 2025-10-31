#include "layer_detect.h"
#include <math.h>
#include "iou.h"
#include "utils.h"
#include "xallocs.h"
#include "image.h"
#include "activations.h"
#include "blas.h"


#define COORD_MULTI 5.0F
#define NO_OBJ_MULTI 0.1F


void cull_predictions_and_do_nms(layer* l, network* net);
void draw_detections(bbox** dets, size_t n_dets, image* img, float thresh);
void apply_grid_scaling(layer* l, network* net);
void pprint_detect_array(float* data, size_t rows, size_t cols, size_t n_classes, size_t n_anchors);


void forward_detect(layer* l, network* net) {
	apply_grid_scaling(l, net);
	float scale_wh = l->scale_grid * 2;
	size_t batch_size = net->batch_size;
	float* p = l->output;  // predictions
	det_sample** samples = net->data.detr.current_batch;
	bbox* anchors = l->anchors;
	bbox* all_anchors = net->anchors;
	int l_id = l->id;
	size_t n_all_anchors = net->n_anchors;
	size_t l_w = l->w;
	size_t l_h = l->h;
	size_t l_wh = l_w * l_h;
	size_t l_n = l->n;
	size_t n_classes = l->n_classes;
	size_t n_anchors = l->n_anchors;
	float cell_size = 1.0F / (float)l_w;
	size_t A = (NUM_ANCHOR_PARAMS + n_classes) * l_wh;
	float* errors = l->errors;
	zero_array(errors, l_n * batch_size);
	float* grads = l->grads;
	zero_array(grads, l_n * batch_size);
	float obj_loss = l->obj_loss;
	float cls_loss = l->cls_loss;
	float iou_loss = l->iou_loss;
	float ignore_thresh = l->ignore_thresh;
	float iou_thresh = l->iou_thresh;
	float obj_normalizer = l->obj_normalizer;
	float max_box_grad = l->max_box_grad;
	int obj_smooth = l->objectness_smooth;
	iou_loss = 0.0F;
	size_t n_iou_loss = 0;
	for (size_t b = 0; b < batch_size; b++) {
		det_sample* sample = samples[b];
		bbox* tboxes = sample->bboxes;  // truth boxes
		size_t n_tboxes = sample->n;
		for (size_t s = 0; s < l_wh; s++) {
			float row = s / l_w;
			float col = s % l_w;
			for (size_t a = 0; a < n_anchors; a++) {
				bbox* anchor = &anchors[a];

				size_t p_index = b * l_n + s + a * A;  // index of prediction "entry"
				size_t obj_index = p_index + l_wh * 4;  // index of objectness score
				float p_obj = p[obj_index];
				
				float w = powf(p[p_index], 2.0F) * scale_wh * anchor->w;
				float h = powf(p[p_index + l_wh], 2.0F) * scale_wh * anchor->h;
				float cx = (p[p_index + l_wh * 2] + col) / l_w;  // predicted value for cx, cy is % of cell size
				float cy = (p[p_index + l_wh * 3] + row) / l_h;
				//printf("w: %f, h: %f, cx: %f, cy: %f\n", p[p_index], p[p_index + l_wh], p[p_index + l_wh * 2], p[p_index + l_wh * 3]);
				bbox pbox = { 0 };  // prediction box
				pbox.cx = cx;
				pbox.cy = cy;
				pbox.w = w;
				pbox.h = h;
				pbox.area = w * h;
				pbox.left = cx - (w / 2.0F);
				pbox.right = pbox.left + w;
				pbox.top = cy - (h / 2.0F);
				pbox.bottom = pbox.top + h;

				size_t cls_index = p[obj_index + l_wh];
				float best_iou = 0.0F;
				// I think this check, when used with obj_smooth=1, is to provide a positive training signal to predictions that appear to be based on learned class features and not just "luck"
				// Also even if objectness gets penalized here, the iou will get improved if the associated anchor is the best, meaning it will eventually have an iou > ignore_thresh
				for (size_t k = 0; k < n_classes; k++) {
					if (p[cls_index + k * l_wh] > 0.25F) {  // darknet uses 0.25
						for (size_t t = 0; t < n_tboxes; t++) {
							float iou = get_iou(pbox, tboxes[t]);
							if (iou > best_iou) best_iou = iou;
						}
						break;
					}
				}
				grads[obj_index] = obj_normalizer * (0.0F - p_obj);  // default objectness gradient
				if (best_iou > ignore_thresh) {
					if (obj_smooth) grads[obj_index] = obj_normalizer * (best_iou - p_obj);
					else grads[obj_index] = 0.0F;
				}
			}
		}
		for (size_t t = 0; t < n_tboxes; t++) {
			bbox tbox = tboxes[t];
			bbox tbox_shifted = tbox;
			tbox_shifted.cx = 0.0F;
			tbox_shifted.cy = 0.0F;
			float best_iou = 0.0F;
			size_t l_a = 0;
			// check which anchor has best "possible" overlap with truthbox (the overlap if anchor and tbox centers were equal)
			for (size_t a = 0; a < n_all_anchors; a++) {
				bbox anchor = all_anchors[a];
				float iou = get_ciou(anchor, tbox_shifted);
				if (anchor.lbl == l_id) l_a++;
				if (iou > best_iou) {
					best_iou = iou;
					if (anchor.lbl != l_id) l_a = 0;
				}
			}
			size_t col = tbox.cx * l_w;
			size_t row = tbox.cy * l_h;
			size_t s = row * l_w + col;
			float cell_left = (float)col * cell_size;
			float cell_top = (float)row * cell_size;
			if (l_a) {  // if best tbox_shifted iou is with an anchor from this layer
				l_a--;
				bbox* anchor = &anchors[l_a];
				size_t p_index = b * l_n + s + l_a * A;  // index of prediction "entry"
				float w = powf(p[p_index], 2.0F) * scale_wh * anchor->w;
				float h = powf(p[p_index + l_wh], 2.0F) * scale_wh * anchor->h;
				float cx = p[p_index + l_wh * 2] + cell_left;  // predicted value for cx, cy is % of cell size
				float cy = p[p_index + l_wh * 3] + cell_top;
				bbox pbox = { 0 };  // prediction box
				pbox.cx = cx;
				pbox.cy = cy;
				pbox.w = w;
				pbox.h = h;
				pbox.area = w * h;
				pbox.left = cx - (w / 2.0F);
				pbox.right = pbox.left + w;
				pbox.top = cy - (h / 2.0F);
				pbox.bottom = pbox.top + h;
				float* dL_dw = &grads[p_index];
				float* dL_dh = &grads[p_index + l_wh];
				float* dL_dx = &grads[p_index + l_wh * 2];
				float* dL_dy = &grads[p_index + l_wh * 3];
				float ciou_loss = get_grads_ciou(pbox, tbox, dL_dx, dL_dy, dL_dw, dL_dh, max_box_grad);
				printf("dL_dw: %f dL_dh: %f dL_dx: %f dL_dy: %f\n", *dL_dw, *dL_dh, *dL_dx, *dL_dy);
				iou_loss += ciou_loss;
				n_iou_loss++;

				size_t obj_index = p_index + l_wh * 4;  // index of objectness score
				float p_obj = p[obj_index];
				if (obj_smooth) {
					if (grads[obj_index] == 0.0F) grads[obj_index] = obj_normalizer * (1.0F - p_obj);
				}
				else grads[obj_index] = 1.0F - p_obj;

				size_t cls_index = p_index + l_wh * NUM_ANCHOR_PARAMS;
				for (size_t k = 0; k < n_classes; k++) {
					float t_cls = (tbox.lbl == k) ? 1.0F : 0.0F;
					grads[cls_index + k * l_wh] = t_cls - p[cls_index + k * l_wh];
				}
				l_a++;
			}
			for (size_t a = 0; a < n_anchors; a++) {
				if (a + 1 == l_a) continue;  // skip if anchor index 'a' was already calculated above
				bbox* anchor = &anchors[a];
				float iou = get_iou(*anchor, tbox_shifted);
				if (iou > iou_thresh) {
					size_t p_index = b * l_n + s + a * A;  // index of prediction "entry"
					float w = p[p_index] * p[p_index] * scale_wh * anchor->w;
					float h = p[p_index + l_wh] * p[p_index + l_wh] * scale_wh * anchor->h;
					float cx = p[p_index + l_wh * 2] + cell_left;  // predicted value for cx, cy is % of cell size
					float cy = p[p_index + l_wh * 3] + cell_top;
					bbox pbox = { 0 };  // prediction box
					pbox.cx = cx;
					pbox.cy = cy;
					pbox.w = w;
					pbox.h = h;
					pbox.area = w * h;
					pbox.left = cx - (w / 2.0F);
					pbox.right = pbox.left + w;
					pbox.top = cy - (h / 2.0F);
					pbox.bottom = pbox.top + h;
					float* dL_dw = &grads[p_index];
					float* dL_dh = &grads[p_index + l_wh];
					float* dL_dx = &grads[p_index + l_wh * 2];
					float* dL_dy = &grads[p_index + l_wh * 3];
					float ciou_loss = get_grads_ciou(pbox, tbox, dL_dx, dL_dy, dL_dw, dL_dh, max_box_grad);
					//printf("dL_dw: %f dL_dh: %f dL_dx: %f dL_dy: %f\n", *dL_dw, *dL_dh, *dL_dx, *dL_dy);
					
					iou_loss += ciou_loss;
					n_iou_loss++;

					size_t obj_index = p_index + l_wh * 4;  // index of objectness score
					float p_obj = p[obj_index];
					if (obj_smooth) {
						float obj_grad = obj_normalizer * (1.0F - p_obj);
						if (grads[obj_index] == 0.0F) grads[obj_index] = obj_grad;
					}
					else grads[obj_index] = 1.0F - p_obj;

					size_t cls_index = p_index + l_wh * NUM_ANCHOR_PARAMS;
					for (size_t k = 0; k < n_classes; k++) {
						float t_cls = (tbox.lbl == k) ? 1.0F : 0.0F;
						grads[cls_index + k * l_wh] = t_cls - p[cls_index + k * l_wh];
					}
				}
			}
		}
	}

	// Just copied from darknet to remind myself to add this functionality
	if (l.iou_thresh < 1.0f) {
		// averages the deltas obtained by the function: delta_yolo_box()_accumulate
		for (j = 0; j < l.h; ++j) {
			for (i = 0; i < l.w; ++i) {
				for (n = 0; n < l.n; ++n) {
					int obj_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 4);
					int box_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 0);
					int class_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 4 + 1);
					const int stride = l.w * l.h;

					if (l.delta[obj_index] != 0)
						averages_yolo_deltas(class_index, box_index, stride, l.classes, l.delta);
				}
			}
		}
	}

	size_t obj_offset = 4 * l_wh;
	size_t b;
#pragma omp parallel for reduction(+:obj_loss) firstprivate(obj_offset, l_n, l_wh, n_anchors)
	for (b = 0; b < batch_size; b++) {
		size_t bn = b * l_n;
		for (size_t s = 0; s < l_wh; s++) {
			size_t bns = bn + s;
			for (size_t a = 0; a < n_anchors; a++) {
				size_t obj_index = bns + a * A + obj_offset;
				obj_loss += powf(grads[obj_index], 2.0F);
			}
		}
	}
	size_t cls_offset = obj_offset + l_wh;
#pragma omp parallel for reduction(+:cls_loss) firstprivate(cls_offset, l_n, l_wh, n_anchors, n_classes)
	for (b = 0; b < batch_size; b++) {
		size_t bn = b * l_n;
		for (size_t s = 0; s < l_wh; s++) {
			size_t bns = bn + s;
			for (size_t a = 0; a < n_anchors; a++) {
				size_t cls_index = bns + a * A + cls_offset;
				for (size_t k = 0; k < n_classes; k++) {
					cls_loss += powf(grads[cls_index + k * l_wh], 2.0F);
				}
			}
		}
	}
	l->obj_loss = obj_loss / (batch_size * l_wh * n_anchors);
	l->cls_loss = cls_loss / (batch_size * l_wh * n_anchors * n_classes);
	if (n_iou_loss) l->iou_loss = iou_loss / (float)n_iou_loss;
	else l->iou_loss = 0.0F;
	l->loss = l->obj_loss + l->cls_loss + l->iou_loss;
	printf("total detect loss: %f\navg obj loss: %f\navg class loss: %f\navg iou loss: %f\n", l->loss, l->obj_loss, l->cls_loss, l->iou_loss);
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
	float obj_thresh = l->nms_obj_thresh;
	float cls_thresh = l->nms_cls_thresh;
	float iou_thresh = l->nms_iou_thresh;
	size_t batch_size = 2;
	size_t l_w = l->w;
	size_t l_h = l->h;
	size_t l_wh = l_w * l_h;
	size_t l_n = l->n;
	size_t n_classes = l->n_classes;
	bbox* anchors = l->anchors;
	size_t n_anchors = l->n_anchors;
	float cell_size = (float)l_w / (float)net_w;
	size_t A = (NUM_ANCHOR_PARAMS + n_classes) * l_wh;
	float* p = l->output;
	size_t ndets = 0;  // debugging
	for (size_t b = 0; b < batch_size; b++) {
		bbox* dets = l->detections;
		bbox** sorted = l->sorted;
		size_t n_dets = 0;
		// cull predictions below thresholds
		for (size_t s = 0; s < l_wh; s++) {
			float row = s / l_w;
			float col = s % l_w;
			float cell_left = col * cell_size;
			float cell_top = row * cell_size;
			for (size_t a = 0; a < n_anchors; a++) {
				size_t p_index = b * l_n + s + a * A;
				size_t obj_index = p_index + l_wh * 4;
				if (p[obj_index] < obj_thresh) continue;  // objectness thresh
				int best_cls = -1;
				float best_cls_score = 0.0F;
				for (size_t i = 0; i < n_classes; i++) {
					size_t cls_index = obj_index + (i + 1) * l_wh;
					if (p[cls_index] > cls_thresh) {  // class confidence thresh
						if (p[cls_index] > best_cls_score) {
							best_cls_score = p[cls_index];
							best_cls = (int)i;
						}
					}
				}
				if (best_cls < 0) continue;
				bbox* anchor = &anchors[a];
				float w = p[p_index] * p[p_index] * 4.0F * anchor->w;  // idk why w & h get multiplied by 4, havent read anything that says to do this but it's what darknet does
				float h = p[p_index + l_wh] * p[p_index + l_wh] * 4.0F * anchor->h;
				float cx = p[p_index + l_wh * 2] + cell_left;  // predicted value for cx, cy is % of cell size
				float cy = p[p_index + l_wh * 3] + cell_top;
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
					if (iou > iou_thresh) {
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
	load_image_to_buffer(samples[1]->imgpath, &img, 1);
	net->draw_thresh = 0.5F;  // TODO: Make a cfg parameter
	draw_detections(l->sorted, ndets, &img, net->draw_thresh);
	////write_image(img, "D:\\TonyDev\\Nardenet\\data\\detector\\test.png");
	show_image(&img);
	//xfree(&img);
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

void apply_grid_scaling(layer* l, network* net) {
	float* p = l->in_layers[0]->output;
	float* q = l->output;
	size_t batch_size = net->batch_size;
	size_t l_n = l->n;
	size_t l_wh = l->w * l->h;
	size_t n_classes = l->n_classes;
	size_t n_anchors = l->n_anchors;
	size_t A = (NUM_ANCHOR_PARAMS + n_classes) * l_wh;
	float alpha = l->scale_grid;
	float beta = (alpha - 1.0F) * 0.5F;
	size_t b;
#pragma omp parallel for firstprivate(l_wh, n_anchors, l_n, A, alpha, beta)
	for (b = 0; b < batch_size; b++) {
		for (size_t s = 0; s < l_wh; s++) {
			for (size_t a = 0; a < n_anchors; a++) {
				size_t p_index = b * l_n + s + a * A;
				q[p_index] = p[p_index];  // w
				q[p_index + l_wh] = p[p_index + l_wh];  // h
				q[p_index + 2 * l_wh] = p[p_index + 2 * l_wh] * alpha - beta;  // cx
				q[p_index + 3 * l_wh] = p[p_index + 3 * l_wh] * alpha - beta;  // cy
				q[p_index + 4 * l_wh] = p[p_index + 4 * l_wh];  // objectness
				for (size_t k = 0; k < n_classes; k++) {
					q[p_index + NUM_ANCHOR_PARAMS * l_wh + k * l_wh] = p[p_index + NUM_ANCHOR_PARAMS * l_wh + k * l_wh];
				}
			}
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