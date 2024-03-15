#ifndef NARDENET_H
#define NARDENET_H

#include "utils.h"


#ifdef __cplusplus
extern "C" {
#endif

    
    typedef struct matrix matrix;
    typedef struct image image;
    typedef struct layer layer;
    typedef struct network network;
    typedef struct net_state net_state;

    typedef enum LAYER_TYPE {
        CONVOLUTIONAL,
        MAXPOOL,
        SOFTMAX,
        YOLO
    } LAYER_TYPE;

    typedef enum ACTIVATION {
        RELU,
        MISH
    } ACTIVATION;

    typedef enum LR_POLICY {
        LR_STEPS
    } LR_POLICY;

    typedef struct matrix {
        size_t rows;
        size_t cols;
        float** vals;
    } matrix;

    typedef struct image {
        size_t w;
        size_t h;
        size_t c;
        float* data;
    } image;

    typedef struct layer {
        LAYER_TYPE type;
        ACTIVATION activation;
        void(*forward)   (struct layer, struct net_state);
        void(*backward)  (struct layer, struct net_state);
        void(*update)    (struct layer, int, float, float, float); //layer, batch, learning rate, momentum, decay
        size_t batch_size;
        size_t w;
        size_t h;
        size_t c;
        size_t out_w;
        size_t out_h;
        size_t out_c;
        size_t n_filters;
        size_t filter_size;
        size_t stride;
        size_t padding;
        size_t n_inputs;
        size_t n_outputs;
        float* output;
        size_t n_weights; // = c * n_filter * filter_size * filter_size;
        float* weights;
        float* biases;
        float* delta;
        float* means;
        float* variances;
        float* losses;
        int train;

    } layer;

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
        float* saturation;
        float* exposure;
        float* hue;
        layer* layers;

    } network;

    typedef struct net_state {
        float* truth;
        float* input;
        float* delta;
        float* workspace;
        int train;
        size_t index;
    } net_state;

#ifdef __cplusplus
}
#endif
#endif
