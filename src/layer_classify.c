#include "layer_classify.h"
#include <stdio.h>
#include "utils.h"
#include "xcuda.h"


void forward_classify(layer* l, network* net) {
	if (net->training) {
		size_t batch_size = net->batch_size;
		size_t n = l->n;
		l->get_loss(l, net);
		printf("Avg class loss: %f\n", l->loss);
		for (size_t b = 0; b < batch_size; b++) {
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
		size_t batch_size = net->batch_size;
		size_t n = l->n;
		l->get_loss(l, net);
		sum_array_gpu(l->gpu.errors, (int)n, l->gpu.loss);
		CUDA_MEMCPY_D2H(&l->loss, l->gpu.loss, sizeof(float));
		float avg_loss = l->loss / (float)batch_size;
		l->loss = avg_loss;
		print_gpu_float_array(l->gpu.output, n * batch_size);
		printf("Avg class loss: %f\n", avg_loss);
		printf("final pointer = %p\n", l->in_layers[0]->gpu.output);
		CUDA_MEMCPY_D2H(l->output, l->gpu.output, n * batch_size * sizeof(float));
		CUDA_MEMCPY_D2H(l->truth, l->gpu.truth, n * batch_size * sizeof(float));
		for (size_t b = 0; b < batch_size; b++) {
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
#else
#pragma warning (suppress:4100)
void forward_classify_gpu(layer* l, network* net) {
	gpu_not_defined();
}
#endif