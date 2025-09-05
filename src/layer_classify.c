#include "layer_classify.h"
#include <stdio.h>
#include "utils.h"
#include "xcuda.h"


void forward_classify_gpu(layer* l, network* net) {
	if (net->training) {
		l->get_loss(l, net);
		float avg_loss = sum_array_gpu(l->errors, l->n) / (float)net->batch_size;
		printf("Avg class loss:      %f\n", avg_loss);
		l->loss = avg_loss;
	}
	else {
		print_top_class_name(l->output, l->n, net->class_names, 1, 1);
	}
}

void forward_classify(layer* l, network* net) {
	if (net->training) {
		l->get_loss(l, net);
		printf("Avg class loss:      %f\n", l->loss);
	}
	else {
		print_top_class_name(l->output, l->n, net->class_names, 1, 1);
	}
}