#include "layer_classify.h"
#include <stdio.h>
#include "utils.h"
#include "xcuda.h"


void forward_classify(layer* l, network* net) {
	if (net->training) {
		l->get_loss(l, net);
		printf("Avg class loss: %f\n", l->loss);
		size_t n = net->n_classes;
		for (size_t b = 0; b < net->batch_size; b++) {
			size_t offset = b * n;
			print_top_class_name(&l->truth[offset], n, net->class_names, 0, 0);
			printf(" : ");
			print_top_class_name(&l->output[offset], n, net->class_names, 1, 1);
		}
	}
	else {
		print_top_class_name(l->output, l->n, net->class_names, 1, 1);
	}
}

#ifdef GPU
void forward_classify_gpu(layer* l, network* net) {
	if (net->training) {
		l->get_loss(l, net);
		float avg_loss = sum_array_gpu(l->errors, (int)l->n) / (float)net->batch_size;
		printf("Avg class loss: %f\n", avg_loss);
		l->loss = avg_loss;
		size_t n = l->n;
		for (size_t b = 0; b < net->batch_size; b++) {
			size_t offset = b * n;
			print_top_class_name(&l->truth[offset], net->n_classes, net->class_names, 0, 0);
			printf(" : ");
			print_top_class_name(&l->output[offset], net->n_classes, net->class_names, 1, 1);
		}
	}
	else {
		print_top_class_name(l->output, l->n, net->class_names, 1, 1);
	}
}
#else
#pragma warning (suppress:4100)
void forward_classify_gpu(layer* l, network* net) {
	gpu_not_defined();
}
#endif