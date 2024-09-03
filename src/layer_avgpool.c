#include "layer_avgpool.h"
#include "derivatives.h"
#include "utils.h"


void forward_avgpool(layer* l, network* net) {
	float* l_Z = l->Z;
	size_t w = l->w;
	size_t h = l->h;
	size_t wh = w * h;
	size_t out_n = l->out_n;
	size_t batch_size = net->batch_size;
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t a = 0; a < l->in_ids.n; a++) {
			layer* inl = l->in_layers[a];
			float* inl_output = inl->output;
			size_t inl_out_c = inl->out_c;
			size_t ch;
#pragma omp parallel for firstprivate(inl_output, wh)
			for (ch = 0; ch < inl_out_c; ch++) {
				float sum = 0.0F;
				size_t ch_offset = ch * wh;
				for (size_t s = 0; s < wh; s++) {
					sum += inl_output[ch_offset + s];
				}
				l_Z[ch] = sum / wh;
			}
			l_Z += inl_out_c;
		}
	}
	if (l->activation) l->activate(l_Z, l->output, l->out_n, net->batch_size);
	if (net->training) zero_array(l->grads, l->out_n * batch_size);
}

void backward_avgpool(layer* l, network* net) {
	size_t batch_size = net->batch_size;
	float* grads = l->grads;
	if (l->activation) {
		if (l->activation == ACT_MISH) get_grads_mish(grads, l->act_inputs, l->out_n, batch_size);  // dC/da * da/dz
		else if (l->activation == ACT_RELU) get_grads_relu(grads, l->act_inputs, l->out_n, batch_size);
		else if (l->activation == ACT_LEAKY) get_grads_leaky_relu(grads, l->act_inputs, l->out_n, batch_size);
		else if (l->activation == ACT_SIGMOID) get_grads_sigmoid(grads, l->output, l->out_n, batch_size);
		else if (l->activation == ACT_TANH) get_grads_tanh(grads, l->act_inputs, l->out_n, batch_size);
		else {
			printf("Incorrect or unsupported activation function.\n");
			exit(EXIT_FAILURE);
		}
	}
	size_t N = batch_size * l->out_n;
	for (size_t a = 0; a < l->in_ids.n; a++) {
		float* inl_grads = l->in_layers[a]->grads;
		size_t i;
#pragma omp parallel for
		for (i = 0; i < N; i++) {
			inl_grads[i] += grads[i];
		}
	}
}