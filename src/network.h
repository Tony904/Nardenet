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
    typedef enum LOSS_TYPE LOSS_TYPE;
    typedef enum REGULARIZATION REGULARIZATION;
    typedef enum NET_TYPE NET_TYPE;

    network* new_network(size_t num_of_layers);
    void build_network(network* net);
    void update_none(layer* l, network* net);

    void free_network(network* net);

    void print_layers(network* net);
    void print_layer(layer* l);
    void print_network(network* net);
    void print_lrpolicy(LR_POLICY lrp);
    void print_layertype(LAYER_TYPE lt);
    void print_activation(ACTIVATION a);
    void print_loss_type(LOSS_TYPE c);
    void print_all_network_weights(network* net);
    void print_some_weights(layer* l, size_t n);
    void print_top_class_name(float* probs, int n_classes, char** class_names);
    void print_network_summary(network* net, int print_training_params);

    typedef struct network {
        size_t n_layers;
        size_t w;
        size_t h;
        size_t c;
        size_t n_classes;
        char** class_names;
        int training;  // 1 = training, 0 = inference only
        size_t batch_size;  // number of images per batch
        size_t subbatch_size;  // number of images per sub-batch, must divide evenly into batch_size
        size_t max_iterations;  // maximum number of training iterations before automatically ending training
        float learning_rate;
        float current_learning_rate;
        LR_POLICY lr_policy;
        floatarr step_percents;
        floatarr step_scaling;
        size_t ease_in;
        float momentum;
        REGULARIZATION regularization;
        void(*reg_loss)             (network*);
        void(*regularize_weights)   (float*, float*, size_t, float);
        int batch_norm;  // 1 if any layer has batch norm enabled
        float loss;
        float decay;
        size_t iteration;  // current iteration, one iteration = one batch processed
        size_t epoch_size;  // number of batches to complete one epoch
        float saturation[2];
        float exposure[2];
        float hue[2];
        layer* input;
        layer* layers;
        floatarr workspace;
        char* cfg_file;
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
        void(*activate)  (float*, float*, size_t, size_t);
        void(*forward)   (layer*, network*);
        void(*backward)  (layer*, network*);
        void(*update)    (layer*, network*);
        void(*get_loss)  (layer*);
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
        float* output;

        floatarr weights;  // n_weights = ksize * ksize * c * n_filters
        float* biases;
        float* act_inputs;  // activation function inputs

        int batch_norm;
        float* Z;  // results of weights * inputs (when batchnorm enabled)
        float* Z_norm;  // Normalized Z
        float* means;
        float* variances;
        float* gammas;  // normalized values scales
        float* rolling_means;
        float* rolling_variances;
        float* mean_grads;
        float* variance_grads;
        float* gamma_grads;

        float* errors;
        float* grads;  // storage for propagated gradients
        float* weight_grads;
        float* bias_grads;
        float* weights_velocity;  // momentum adjustment for weights
        float* biases_velocity;  // momentum adjustment for biases

        float loss;
        LOSS_TYPE loss_type;

        intarr in_ids;
        intarr out_ids;
        layer** in_layers;

        int train;

        size_t n_classes;
        size_t n_anchors;
        size_t* anchors;
        float* truth;

        float** maxpool_addresses;  // addresses of input layer outputs that were max values (for backprop)
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

    typedef enum LOSS_TYPE {
        LOSS_NONE,
        LOSS_MSE,  // mean squared error
        LOSS_BCE,  // binary cross-entropy, used for 2-class classification (i.e. good/bad, yes/no)
        LOSS_CCE  // categorical cross-entropy, used for >2 class classification
    } LOSS_TYPE;

    typedef enum REGULARIZATION {
        REG_NONE,
        REG_L1,  // sum of absolute values of weights
        REG_L2  // sum of squared values of weights (much preferred over L1)
    } REGULARIZATION;

#ifdef __cplusplus
}
#endif
#endif