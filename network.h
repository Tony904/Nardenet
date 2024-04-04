#ifndef NETWORK_H
#define NETWORK_H

#include "xarrays.h"
#include "image.h"
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
    void build_network(network* net);
    void free_network(network* net);
    void print_layers(layer* l, size_t num_of_layers);
    void print_layer(layer* l);
    void print_network(network* net);
    void print_lrpolicy(LR_POLICY lrp);
    void print_layertype(LAYER_TYPE lt);
    void print_activation(ACTIVATION a);

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
        size_t iteration;  // current iteration, one iteration = one batch processed
        size_t epoch_size;  // number of batches to complete one epoch
        float saturation[2];
        float exposure[2];
        float hue[2];
        image* input;
        layer* layers;
        float* output;
        int i;  // current layer index
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
        size_t n_filters;
        size_t ksize;
        size_t stride;
        size_t pad;
        size_t w;
        size_t h;
        size_t c;
        size_t n; // n_inputs = w * h * c
        size_t out_w; // out_w = ((w + (pad * 2) - ksize) / stride) + 1
        size_t out_h; // out_h = ((h + (pad * 2) - ksize) / stride) + 1
        size_t out_c; // out_c = n_filters
        size_t out_n; // out_n = out_w * out_h * out_c
        float* output;  // n_outputs = out_w * out_h * out_c
        floatarr weights;  // n_weights = ksize * ksize * c * n_filters
        float* biases;
        float* delta;  // predicted - truth (?)
        float* means;
        float* variances;
        float* losses;
        intarr in_ids;
        intarr out_ids;
        layer** in_layers;
        int train;
        int batch_norm;
        size_t* anchors;
    } layer;

    typedef enum LAYER_TYPE {
        NONE_LAYER,
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