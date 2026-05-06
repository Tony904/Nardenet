#include "layer_classify.h"
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "xcuda.h"
#include "loss.h"


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
		if (l->loss < 0.1F || isnan(l->loss)) {
			printf("\n[DONE]\n");
			net->abort = 1;
		}
	}
	else {
		print_top_class_name(l->output, l->n, net->class_names, 1, 1);
	}
}

void forward_classify_cpu_gpu_compare(layer* l, network* net) {
	size_t batch_size = net->batch_size;

	float* errors_cpu = l->errors;
	float* output_cpu = l->output;
	float* grads_cpu = l->grads;
	float* truth_cpu = l->truth;

	float* errors_gpu = l->gpu.errors;
	float* output_gpu = l->gpu.output;
	float* grads_gpu = l->gpu.grads;
	float* truth_gpu = l->gpu.truth;

	size_t size = l->n * batch_size;
	compare_cpu_gpu_arrays(errors_cpu, errors_gpu, size, l->id, "forward classify, errors, pre-loss");
	compare_cpu_gpu_arrays(output_cpu, output_gpu, size, l->id, "forward classify, output, pre-loss");
	compare_cpu_gpu_arrays(grads_cpu, grads_gpu, size, l->id, "forward clsasify, grads, pre-loss");
	compare_cpu_gpu_arrays(truth_cpu, truth_gpu, size, l->id, "forward classify, truth, pre-loss");

	loss_cce(l, net);
	printf("AVG CLASS LOSS CPU: %f\n", l->loss);
	
	loss_cce_gpu(l, net);
	sum_array_gpu(l->gpu.errors, (int)(l->n * batch_size), l->gpu.loss);
	CUDA_MEMCPY_D2H(&l->loss, l->gpu.loss, sizeof(float));
	float avg_loss_gpu = l->loss / (float)batch_size;
	printf("AVG CLASS LOSS GPU: %f\n", avg_loss_gpu);

	compare_cpu_gpu_arrays(errors_cpu, errors_gpu, size, l->id, "forward classify, errors, post-loss");
	compare_cpu_gpu_arrays(output_cpu, output_gpu, size, l->id, "forward classify, output, post-loss");
	compare_cpu_gpu_arrays(grads_cpu, grads_gpu, size, l->id, "forward clsasify, grads, post-loss");
	compare_cpu_gpu_arrays(truth_cpu, truth_gpu, size, l->id, "forward classify, truth, post-loss");

	float min_loss = fminf(avg_loss_gpu, l->loss);
	if (min_loss < 0.1F || isnan(avg_loss_gpu)) {
		printf("\n[DONE]\n");
		wait_for_key_then_exit();
	}
}

#ifdef GPU
void forward_classify_gpu(layer* l, network* net) {
	if (net->training) {
		size_t batch_size = net->batch_size;
		size_t n = l->n;
		l->get_loss(l, net);
		sum_array_gpu(l->gpu.errors, (int)(n * batch_size), l->gpu.loss);
		CUDA_MEMCPY_D2H(&l->loss, l->gpu.loss, sizeof(float));
		float avg_loss = l->loss / (float)batch_size;
		l->loss = avg_loss;
		printf("Avg class loss: %f\n", avg_loss);
		CUDA_MEMCPY_D2H(l->output, l->gpu.output, n * batch_size * sizeof(float));
		CUDA_MEMCPY_D2H(l->truth, l->gpu.truth, n * batch_size * sizeof(float));
		for (size_t b = 0; b < batch_size; b++) {
			size_t offset = b * n;
			print_top_class_name(&l->truth[offset], n, net->class_names, 0, 0);
			printf(" : ");
			print_top_class_name(&l->output[offset], n, net->class_names, 1, 1);
		}
		if (avg_loss < 0.1F || isnan(avg_loss)) {
			printf("\n[DONE]\n");
			net->abort = 1;
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