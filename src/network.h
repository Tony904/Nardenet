#ifndef NETWORK_H
#define NETWORK_H

#include <stdlib.h>
#include "xarrays.h"
#include "data.h"

#define NUM_ANCHOR_PARAMS 5  // probability contains any object, center_x, center_y, width, height

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct network network;
    typedef struct layer layer;
    typedef enum LR_POLICY LR_POLICY;
    typedef enum LAYER_TYPE LAYER_TYPE;
    typedef enum ACTIVATION ACTIVATION;
    typedef enum COST_TYPE COST_TYPE;
    typedef enum NET_TYPE NET_TYPE;

    network* new_network(size_t num_of_layers);
    void build_network(network* net);
    void free_network(network* net);
    void print_layers(network* net);
    void print_layer(layer* l);
    void print_network(network* net);
    void print_lrpolicy(LR_POLICY lrp);
    void print_layertype(LAYER_TYPE lt);
    void print_activation(ACTIVATION a);
    void print_cost_type(COST_TYPE c);

    typedef struct network {
        size_t n_layers;
        size_t w;
        size_t h;
        size_t c;
        size_t n_classes;
        char** class_names;
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
        float* anchors;
        size_t n_anchors;
        layer* input;
        layer* layers;
        float* output;
        float* grads;
        floatarr workspace;
        char* dataset_dir;
        char* weights_file;
        char* backup_dir;
        NET_TYPE type;
        union data {
            classifier_dataset clsr;
            detector_dataset detr;
        } data;
    } network;

    typedef struct layer {
        int id;
        LAYER_TYPE type;
        ACTIVATION activation;
        void(*activate)  (layer*);
        void(*forward)   (layer*);
        void(*backprop)  (layer*, network*);
        void(*update)    (layer*, int, float, float, float); //layer, batch, learning rate, momentum, decay
        void(*get_cost)  (layer*);
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
        float* act_input;
        float* means;
        float* variances;
        float* errors;
        float* grads;
        float* weight_updates;
        float cost;
        COST_TYPE cost_type;
        intarr in_ids;
        intarr out_ids;
        layer** in_layers;
        int train;
        int batch_norm;
        size_t n_classes;
        size_t* anchors;
        float* truth;
    } layer;

    typedef enum NET_TYPE {
        NET_CLASSIFY,
        NET_DETECT
    } NET_TYPE;

    typedef enum LAYER_TYPE {
        LAYER_INPUT = -1,
        LAYER_NONE,
        LAYER_CONV,
        LAYER_MAXPOOL,
        LAYER_CLASSIFY,
        LAYER_DETECT
    } LAYER_TYPE;

    typedef enum LR_POLICY {
        LR_STEPS
    } LR_POLICY;

    typedef enum ACTIVATION {
        ACT_NONE,
        ACT_RELU,
        ACT_LEAKY,
        ACT_MISH,
        ACT_SIGMOID,
        ACT_SOFTMAX
    } ACTIVATION;

    typedef enum COST_TYPE {
        COST_NONE,
        COST_MSE,  // mean squared error
        COST_BCE,  // binary cross-entropy, used for 2-class classification (i.e. good/bad, yes/no)
        COST_CCE  // categorical cross-entropy, used for >2 class classification
    } COST_TYPE;

#ifdef __cplusplus
}
#endif
#endif