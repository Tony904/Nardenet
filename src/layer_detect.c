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
#define GET_PREDICTION_BOX(p, i, anchor, spatial, scale, left, top) { .w=p[i]*p[i]*scale*anchor->w,.h=p[i+spatial]*p[i+spatial]*scale*anchor->h,.cx=p[i+spatial*2]+left,.cy=p[i+spatial*3]+top }
#define SETUP_BBOX(b) (b).area=(b).w*(b).h;(b).left=(b).cx-(b).w/2.0F;(b).right=(b).cx-(b).w/2.0F+(b).w;(b).top=(b).cy-(b).h/2.0F;(b).bottom=(b).cy-(b).h/2.0F+(b).h

static inline float clamp_x(float x, float thresh) {
	if (x > thresh) return thresh;
	if (x < -thresh) return -thresh;
	return x;
}


void get_class_grads(size_t cls_index, size_t n_classes, float* const grads, const float* const output, bbox truth_box, size_t spatial);
void debug_detections(layer* l, network* net, int waitkey);
void get_detections(layer* l, network* net, size_t b);
void nms(layer* l);
void draw_detections(bbox** dets, size_t n_dets, image* img, float thresh);
void display_detections(layer* l, network* net, float thresh, size_t b, int waitkey);
void apply_grid_scaling(layer* l, network* net);
void pprint_detect_array(float* data, size_t rows, size_t cols, size_t n_classes, size_t n_anchors);



// p = output
void forward_detect_batch(const float* const p, float* const grads,
							const bbox* const anchors, const bbox* const all_anchors,
							const det_sample** const samples,
							float* iou_losses,
							size_t b, int l_id,
							size_t l_w, size_t l_h, size_t l_n, float scale_wh,
							size_t n_anchors, size_t n_all_anchors, size_t n_classes,
							int obj_smooth, float obj_normalizer, float max_box_grad,
							float ignore_thresh, float iou_thresh) {

	float iou_loss = 0.0F;
	size_t n_iou_losses = 0;
	size_t l_wh = l_w * l_h;
	size_t A = (NUM_ANCHOR_PARAMS + n_classes) * l_wh;
	float cell_size = 1.0F / (float)l_wh;
	det_sample sample = *samples[b];
	bbox* tboxes = sample.bboxes;  // truth boxes
	size_t n_tboxes = sample.n;
	for (size_t s = 0; s < l_wh; s++) {
		float row = s / l_w;
		float col = s % l_w;
		float cell_left = row * cell_size;
		float cell_top = col * cell_size;
		for (size_t a = 0; a < n_anchors; a++) {
			const bbox* const anchor = &anchors[a];

			size_t p_index = b * l_n + s + a * A;  // index of prediction "entry"
			bbox pbox = GET_PREDICTION_BOX(p, p_index, anchor, l_wh, scale_wh, cell_left, cell_top);
			SETUP_BBOX(pbox);
			//printf("w: %f, h: %f, cx: %f, cy: %f\n", p[p_index], p[p_index + l_wh], p[p_index + l_wh * 2], p[p_index + l_wh * 3]);

			size_t obj_index = p_index + l_wh * 4;  // index of objectness score
			size_t cls_index = obj_index + l_wh;
			float best_iou = 0.0F;
			// I think this check, when used with obj_smooth=1, is to provide a positive training signal to predictions that appear to be based on learned class features.
			// I think it's also to prevent processing the iou for all truths for anchors that predict no object/class, as that would be expensive.
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
			float p_obj = p[obj_index];
			grads[obj_index] = obj_normalizer * (p_obj);  // default objectness gradient
			if (best_iou > ignore_thresh) { // I believe this is to "throttle" good prediction gradients so that they don't cause stability issues.
				if (obj_smooth) grads[obj_index] = obj_normalizer * (p_obj - best_iou);
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
			float ciou = get_iou(anchor, tbox_shifted);
			if (anchor.lbl == l_id) l_a++;
			if (ciou > best_iou) {
				best_iou = ciou;
				if (anchor.lbl != l_id) l_a = 0;
			}
		}
		size_t col = tbox.cx * l_w;
		size_t row = tbox.cy * l_h;
		size_t s = row * l_w + col;
		float cell_left = ((float)col) * cell_size;
		float cell_top = ((float)row) * cell_size;
		if (l_a) {  // if best tbox_shifted iou is with an anchor from this layer
			l_a--;
			const bbox* const anchor = &anchors[l_a];
			size_t p_index = b * l_n + s + l_a * A;  // index of prediction "entry"
			bbox pbox = GET_PREDICTION_BOX(p, p_index, anchor, l_wh, scale_wh, cell_left, cell_top);
			SETUP_BBOX(pbox);
			float dw = 0.0F;
			float dh = 0.0F;
			float dx = 0.0F;
			float dy = 0.0F;
			float ciou_loss = 1.0F - get_grads_ciou(pbox, tbox, &dx, &dy, &dw, &dh);
			grads[p_index] += clamp_x(-dw, max_box_grad);
			grads[p_index + l_wh] += clamp_x(-dh, max_box_grad);
			grads[p_index + l_wh * 2] += clamp_x(-dx, max_box_grad);
			grads[p_index + l_wh * 3] += clamp_x(-dy, max_box_grad);
			//printf("dL_dw: %f dL_dh: %f dL_dx: %f dL_dy: %f\n", grads[p_index], grads[p_index + l_wh], grads[p_index + l_wh * 2], grads[p_index + l_wh * 3]);
			iou_loss += ciou_loss;
			n_iou_losses++;

			size_t obj_index = p_index + l_wh * 4;  // index of objectness score
			float p_obj = p[obj_index];
			//printf("p_obj = %f\n", p_obj);
			if (obj_smooth) {
				if (grads[obj_index] == 0.0F) grads[obj_index] = obj_normalizer * (p_obj - 1.0F);
			}
			else grads[obj_index] = p_obj - 1.0F;

			size_t cls_index = obj_index + l_wh;
			get_class_grads(cls_index, n_classes, grads, p, tbox, l_wh);
			l_a++;
		}
		for (size_t a = 0; a < n_anchors; a++) {
			if (a + 1 == l_a) continue;  // skip if anchor index 'a' was already calculated above
			const bbox* const anchor = &anchors[a];
			float iou = get_iou(*anchor, tbox_shifted);
			if (iou > iou_thresh) {
				size_t p_index = b * l_n + s + a * A;  // index of prediction "entry"
				bbox pbox = GET_PREDICTION_BOX(p, p_index, anchor, l_wh, scale_wh, cell_left, cell_top);
				SETUP_BBOX(pbox);
				float dw = 0.0F;
				float dh = 0.0F;
				float dx = 0.0F;
				float dy = 0.0F;
				float ciou_loss = 1.0F - get_grads_ciou(pbox, tbox, &dx, &dy, &dw, &dh);
				grads[p_index] += clamp_x(-dw, max_box_grad);
				grads[p_index + l_wh] += clamp_x(-dh, max_box_grad);
				grads[p_index + l_wh * 2] += clamp_x(-dx, max_box_grad);
				grads[p_index + l_wh * 3] += clamp_x(-dy, max_box_grad);
				//printf("dL_dw: %f dL_dh: %f dL_dx: %f dL_dy: %f\n", grads[p_index], grads[p_index + l_wh], grads[p_index + l_wh * 2], grads[p_index + l_wh * 3]);
				iou_loss += ciou_loss;
				n_iou_losses++;

				size_t obj_index = p_index + l_wh * 4;  // index of objectness score
				float p_obj = p[obj_index];
				if (obj_smooth) {
					if (grads[obj_index] == 0.0F) grads[obj_index] = obj_normalizer * (p_obj - 1.0F);
				}
				else grads[obj_index] = p_obj - 1.0F;

				size_t cls_index = obj_index + l_wh;
				get_class_grads(cls_index, n_classes, grads, p, tbox, l_wh);
			}
		}
	}

	if (iou_thresh < 1.0F) {  // iou_thresh being less than 1 is the only scenario where multiple positive training signals for class can occur
		// averages the bbox gradients across classes with positive training signals
		for (size_t s = 0; s < l_wh; s++) {
			for (size_t a = 0; a < n_anchors; a++) {
				size_t p_index = b * l_n + s + a * A;
				size_t obj_index = p_index + l_wh * 4;
				size_t cls_index = p[obj_index + l_wh];
				if (grads[obj_index]) {
					float count = 0.0F;
					for (int k = 0; k < n_classes; k++) {
						if (grads[cls_index + k * l_wh] > 0.0F) count += 1.0F;
					}
					if (count > 0.0F) {
						grads[p_index] /= count;
						grads[p_index + l_wh] /= count;
						grads[p_index + l_wh * 2] /= count;
						grads[p_index + l_wh * 3] /= count;
					}
				}
			}
		}
	}
	if (n_iou_losses == 0) {
		iou_losses[b] = -1.0F; // values < 0 get ignored when total iou loss is calculated later
	}
	else {
		iou_losses[b] = iou_loss / (float)n_iou_losses;
	}
}

void forward_detect(layer* l, network* net) {
	apply_grid_scaling(l, net);
	float scale_wh = l->scale_grid * 2.0F;
	size_t batch_size = net->batch_size;
	const float* const output = l->output;  // predictions
	const det_sample** const samples = net->data.detector.current_batch;
	const bbox* const anchors = l->anchors;
	const bbox* const all_anchors = net->anchors;
	int l_id = l->id;
	size_t n_all_anchors = net->n_anchors;
	size_t l_w = l->w;
	size_t l_h = l->h;
	size_t l_n = l->n;
	size_t n_classes = l->n_classes;
	size_t n_anchors = l->n_anchors;
	float* errors = l->errors;
	zero_array(errors, batch_size);
	float* const grads = l->grads;
	zero_array(grads, l_n * batch_size);
	float ignore_thresh = l->ignore_thresh;
	float iou_thresh = l->iou_thresh;
	float obj_normalizer = l->obj_normalizer;
	float max_box_grad = l->max_box_grad;
	int obj_smooth = l->objectness_smooth;
	size_t b;
#pragma omp parallel for firstprivate(output, grads, anchors, all_anchors, samples, errors, b, l_id, l_w, l_h, l_n, scale_wh, n_anchors, n_all_anchors, n_classes, obj_smooth, obj_normalizer, max_box_grad, ignore_thresh, iou_thresh)
	for (b = 0; b < batch_size; b++) {
		forward_detect_batch(output, grads,
			anchors, all_anchors,
			samples,
			errors,
			b, l_id,
			l_w, l_h, l_n, scale_wh,
			n_anchors, n_all_anchors, n_classes,
			obj_smooth, obj_normalizer, max_box_grad,
			ignore_thresh, iou_thresh);
	}

	size_t l_wh = l_w * l_h;
	size_t A = (NUM_ANCHOR_PARAMS + n_classes) * l_wh;
	// sum objectness loss
	float obj_loss = 0.0F;
	size_t obj_offset = l_wh * 4;
#pragma omp parallel for reduction(+:obj_loss) firstprivate(obj_offset, l_n, l_wh, n_anchors)
	for (b = 0; b < batch_size; b++) {
		size_t bn = b * l_n;
		for (size_t s = 0; s < l_wh; s++) {
			size_t bns = bn + s;
			for (size_t a = 0; a < n_anchors; a++) {
				size_t obj_index = bns + a * A + obj_offset;
				obj_loss += (grads[obj_index] * grads[obj_index]);
			}
		}
	}
	// sum class loss
	float cls_loss = 0.0F;
	size_t cls_offset = obj_offset + l_wh;
#pragma omp parallel for reduction(+:cls_loss) firstprivate(cls_offset, l_n, l_wh, n_anchors, n_classes)
	for (b = 0; b < batch_size; b++) {
		size_t bn = b * l_n;
		float n_tboxes = (float)samples[b]->n;
		for (size_t s = 0; s < l_wh; s++) {
			size_t bns = bn + s;
			for (size_t a = 0; a < n_anchors; a++) {
				size_t cls_index = bns + a * A + cls_offset;
				for (size_t k = 0; k < n_classes; k++) {
					cls_loss += (grads[cls_index + k * l_wh] * grads[cls_index + k * l_wh]) / n_tboxes;
				}
			}
		}
	}

	l->obj_loss = obj_loss / (float)(batch_size * l_wh * n_anchors);
	l->cls_loss = cls_loss / (float)(batch_size * l_wh * n_anchors * n_classes);

	float iou_loss = 0.0F;
	int n_iou_loss = 0;
	for (b = 0; b < batch_size; b++) {
		if (errors[b] < 0.0F) continue;
		iou_loss += errors[b];
		n_iou_loss++;
	}
	l->iou_loss = iou_loss / (float)n_iou_loss;	
	l->loss = l->obj_loss + l->cls_loss + l->iou_loss;
	printf("total detect loss: %f\navg obj loss: %f\navg class loss: %f\navg iou loss: %f\n", l->loss, l->obj_loss, l->cls_loss, l->iou_loss);
	
	for (size_t i = 0; i < l->n; i++) {
		for (b = 0; b < batch_size; b++) {
			size_t index = b * l->n + i;
			printf("grads[%zu] = %f   ", index, grads[index]);
		}
		printf("\n");
	}
	debug_detections(l, net, 1);
}

void get_class_grads(size_t cls_index, size_t n_classes, float* const grads, const float* const output, bbox truth_box, size_t spatial) {
	if (grads[cls_index + spatial * truth_box.lbl]) {
		float grad = output[cls_index + spatial * truth_box.lbl] - 1.0F;
		if (!isnan(grad) && !isinf(grad)) grads[cls_index + spatial * truth_box.lbl] = grad;
		return;
	}
	for (size_t k = 0; k < n_classes; k++) {
		float t_cls = (truth_box.lbl == k) ? 1.0F : 0.0F;
		grads[cls_index + spatial * k] = output[cls_index + spatial * k] - t_cls;
	}
}

// Gets detections and stores them in l->sorted
void get_detections(layer* l, network* net, size_t b) {
	size_t net_w = net->w;
	float cull_thresh = l->cull_thresh;
	float scale = l->scale_grid * 2.0F;
	size_t l_w = l->w;
	size_t l_wh = l_w * l->h;
	size_t l_n = l->n;
	size_t n_classes = l->n_classes;
	bbox* anchors = l->anchors;
	size_t n_anchors = l->n_anchors;
	float cell_size = (float)l_w / (float)net_w;
	size_t A = (NUM_ANCHOR_PARAMS + n_classes) * l_wh;
	float* p = l->output;
	bbox* dets = l->detections;  // note: l->detections size = l->w * l->h * l->n_anchors * sizeof(bbox)
	// cull predictions below thresholds
	size_t s;
#pragma omp parallel for firstprivate(cull_thresh, scale, l_w, l_wh, l_n, n_classes, anchors, n_anchors, cell_size, A, p, dets)
	for (s = 0; s < l_wh; s++) {
		float row = s / l_w;
		float col = s % l_w;
		float cell_left = col * cell_size;
		float cell_top = row * cell_size;
		for (size_t a = 0; a < n_anchors; a++) {
			size_t d_index = l_wh * a + s;
			size_t p_index = b * l_n + s + a * A;
			size_t obj_index = p_index + l_wh * 4;
			if (p[obj_index] < cull_thresh) {
				dets[d_index].prob = 0.0F;
				continue;
			}
			int best_cls = -1;
			float best_cls_score = 0.0F;
			size_t cls_index = obj_index + l_wh;
			for (size_t k = 0; k < n_classes; k++) {
				float cls_score = p[cls_index + k * l_wh];
				if (cls_score > cull_thresh) {  // class confidence thresh
					if (cls_score > best_cls_score) {
						best_cls_score = cls_score;
						best_cls = (int)k;
					}
				}
			}
			if (best_cls < 0) {
				dets[d_index].prob = 0.0F;
				continue;
			}
			bbox* anchor = &anchors[a];
			bbox pbox = GET_PREDICTION_BOX(p, p_index, anchor, l_wh, scale, cell_left, cell_top);
			SETUP_BBOX(pbox);
			if (pbox.left >= 1.0F || pbox.right <= 0.0F || pbox.top >= 1.0F || pbox.bottom <= 0.0F) {
				dets[d_index].prob = 0.0F;
				continue;
			};
			
			bbox* det = &dets[d_index];
			det->prob = best_cls_score;
			det->lbl = best_cls;
			float left = fmaxf(0.0F, pbox.left);
			float right = fminf(1.0F, pbox.right);
			float top = fmaxf(0.0F, pbox.top);
			float bottom = fminf(1.0F, pbox.bottom);
			det->cx = (left + right) / 2.0F;
			det->cy = (top + bottom) / 2.0F;
			det->w = right - left;
			det->h = bottom - top;
			SETUP_BBOX(*det);
		}
	}
	bbox** sorted = l->sorted;
	size_t n_dets = 0;
	for (size_t i = 0; i < l_wh * n_anchors; i++) {
		if (dets[i].prob) {
			*sorted = &dets[i];
			n_dets++;
			sorted++;
		}
	}
	for (size_t i = 0; i < n_dets; i++) {
		printf("sorted[%zu]->prob = %f\n", i, l->sorted[i]->prob);
	}
	l->n_dets = n_dets;
}

void nms(layer* l) {
	bbox** sorted = l->sorted;
	size_t n_dets = l->n_dets;
	// sort remaining detections
	size_t i = 0;
	while (i + 1 < n_dets) {
		if (sorted[i]->prob < sorted[i + 1]->prob) {
			bbox* tmp = sorted[i];
			sorted[i] = sorted[i + 1];
			sorted[i + 1] = tmp;
			i = i ? i - 1 : i + 1;
		}
		else i++;
	}
	// nms
	float iou_thresh = l->nms_iou_thresh;
	size_t dupes = 0;
	for (i = 0; i < n_dets; i++) {
		bbox* test = sorted[i];
		if (test->prob) {
			for (size_t j = i + 1; j < n_dets; j++) {
				if (test->lbl == sorted[j]->lbl) {
					float diou = get_diou(*test, *sorted[j], 0);
					if (diou > iou_thresh) {
						sorted[j]->prob = 0.0F;
						sorted[j]->lbl = -1;
						dupes++;
					}
				}
			}
		}
	}
	printf("Detections suppressed: %zu\n", dupes);
}

void display_detections(layer* l, network* net, float thresh, size_t b, int waitkey) {
	det_sample** samples = net->data.detector.current_batch;
	image* img = load_image(samples[b]->imgpath);
	draw_detections(l->sorted, l->n_dets, img, thresh);
	show_image("test", img, waitkey);
	xfree(&img->data);
	xfree(&img);
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
		if (dets[i]->prob < thresh || dets[i]->lbl < 0) continue;
		bbox* det = dets[i];
		size_t box_left = det->left * img_w;
		size_t box_top = det->top * img_h;
		size_t box_right = (det->right * img_w) - 1.0F;
		size_t box_bottom = (det->bottom * img_h) - 1.0F;
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

void debug_detections(layer* l, network* net, int waitkey) {
	for (size_t b = 0; b < net->batch_size; b++) {
		get_detections(l, net, b);
		nms(l);
		display_detections(l, net, 0.5F, b, waitkey);
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