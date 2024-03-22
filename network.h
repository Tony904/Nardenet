#ifndef NETWORK_H
#define NETWORK_H

#include "xarrays.h"
#include <stdlib.h>


#ifdef __cplusplus
extern "C" {
#endif

    typedef struct network network;
    typedef enum LR_POLICY LR_POLICY;
    typedef struct layer layer;
    typedef enum LAYER_TYPE LAYER_TYPE;
    typedef enum ACTIVATION ACTIVATION;

    network* new_network(size_t num_of_layers);
    void free_network(network* net);
    void print_network(network* net);

    typedef struct network {
        size_t n_layers;
        size_t w;
        size_t h;
        size_t c;
        size_t batch_size;  // number of images per batch
        size_t subbatch_size;  // number of images per sub-batch, must divide evenly into batch_size
        size_t max_iterations;  // maximum number of training iterations before automatically ending training
        float learning_rate;
        LR_POLICY lr_policy;
        floatarr step_percents;
        floatarr step_scaling;
        size_t ease_in;
        float momentum;
        float decay;
        float* output;
        size_t iteration;  // current iteration, one iteration = one batch processed
        size_t epoch_size;  // number of batches to complete one epoch
        float saturation[2];
        float exposure[2];
        float hue[2];
        layer* layers;
    } network;

    typedef enum LR_POLICY {
        LR_STEPS
    } LR_POLICY;

    typedef struct layer {
        int id;
        LAYER_TYPE type;
        ACTIVATION activation;
        void(*forward)   (layer*, network*);
        void(*backward)  (layer*, network*);
        void(*update)    (layer*, int, float, float, float); //layer, batch, learning rate, momentum, decay
        size_t batch_size;
        size_t w;
        size_t h;
        size_t c;
        size_t out_w;
        size_t out_h;
        size_t out_c;
        size_t n_filters;
        size_t k_size;
        size_t stride;
        size_t padding;
        size_t n_inputs;
        size_t n_outputs;
        size_t n_weights; // = c * n_filter * filter_size * filter_size;
        float* output;
        float* weights;
        float* biases;
        float* delta;
        float* means;
        float* variances;
        float* losses;
        size_t* anchors;
        int train;
        int batch_norm;
    } layer;

    typedef enum LAYER_TYPE {
        UNDEF_LAYER = -1,
        NOT_A_LAYER,
        CONV,
        MAXPOOL,
        SOFTMAX,
        YOLO
    } LAYER_TYPE;

    typedef enum ACTIVATION {
        RELU,
        MISH
    } ACTIVATION;

#ifdef __cplusplus
}
#endif
#endif