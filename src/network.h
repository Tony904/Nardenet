#ifndef NETWORK_H
#define NETWORK_H

#include <stdlib.h>
#include "xarrays.h"

#define NUM_ANCHOR_PARAMS 5  // probability contains any object, center_x, center_y, width, height

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct network network;
    typedef struct layer layer;
    typedef struct class_set class_set;
    typedef struct classifier_dataset classifier_dataset;
    typedef struct bbox bbox;
    typedef struct det_cell det_cell;
    typedef struct det_sample det_sample;
    typedef struct detector_dataset detector_dataset;
    typedef enum LR_POLICY LR_POLICY;
    typedef enum LAYER_TYPE LAYER_TYPE;
    typedef enum ACTIVATION ACTIVATION;
    typedef enum LOSS_TYPE LOSS_TYPE;
    typedef enum REGULARIZATION REGULARIZATION;
    typedef enum NET_TYPE NET_TYPE;


    network* new_network(size_t num_of_layers);
    void build_network(network* net);
    void free_network(network* net);

    void get_activation_grads(layer* l, size_t batch_size);
    void get_activation_grads_gpu(layer* l, size_t batch_size);

    void print_layers(network* net);
    void print_layer(layer* l);
    void print_network(network* net);
    void print_lrpolicy(LR_POLICY lrp);
    void print_layertype(LAYER_TYPE lt);
    void print_activation(ACTIVATION a);
    void print_loss_type(LOSS_TYPE c);
    void print_all_network_weights(network* net);
    void print_some_weights(layer* l, size_t n);
    void print_top_class_name(float* probs, size_t n_classes, char** class_names, int include_prob, int new_line);
    void print_network_summary(network* net, int print_training_params);
    void print_network_structure(network* net);
    void print_prediction_results(network* net, layer* prediction_layer);

    typedef struct classifier_dataset {
        class_set* sets;
        size_t n;       // # of sets/classes
        size_t* rands;  // array of non-repeating random numbers of size n
        size_t ri;      // rands index
    } classifier_dataset;

    typedef struct detector_dataset {
        det_sample* samples;
        size_t n;       // # of samples
        size_t* rands;  // array of non-repeating random numbers of size n
        size_t ri;      // rands index
        det_sample** current_batch;
    } detector_dataset;

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
        size_t save_frequency;

        float learning_rate;
        float current_learning_rate;
        LR_POLICY lr_policy;
        floatarr step_percents;
        floatarr step_scaling;
        size_t coswr_frequency;
        float coswr_multi;
        float exp_decay;
        float poly_pow;
        size_t ease_in;
        float momentum;

        REGULARIZATION regularization;
        void(*reg_loss)             (network*);
        void(*regularize_weights)   (float*, float*, size_t, float);
        int batchnorm;  // 1 if any layer has batch norm enabled
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
        char* save_dir;
        NET_TYPE type;
        union data {
            classifier_dataset clsr;
            detector_dataset detr;
        } data;

        bbox* anchors;  // anchors across all detection layers
        size_t n_anchors;

        float draw_thresh;  // threshold for drawing detections (if p >= draw_thresh then draw)
    } network;

    typedef struct layer {
        int id;
        LAYER_TYPE type;
        ACTIVATION activation;
        void(*activate)  (float*, float*, size_t, size_t);
        void(*forward)   (layer*, network*);
        void(*backward)  (layer*, network*);
        void(*update)    (layer*, network*);
        void(*get_loss)  (layer*, network*);
        size_t n_groups;
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

        int batchnorm;
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
        float* gamma_velocities;

        float* errors;
        float* grads;  // storage for propagated gradients
        float* weight_grads;
        float* bias_grads;
        float* weight_velocities;  // momentum adjustment for weights
        float* bias_velocities;  // momentum adjustment for biases

        intarr in_ids;
        layer** in_layers;

        LOSS_TYPE loss_type;
        float* loss_gpu;
        float loss;
        float obj_loss;
        float cls_loss;
        float iou_loss;

        int train;

        size_t n_classes;
        size_t n_anchors;
        bbox* anchors;  // base anchors that will get copied to each cell
        bbox* detections;
        bbox** sorted;
        float nms_obj_thresh;
        float nms_cls_thresh;
        float nms_iou_thresh;
        float* truth;  // truths for classifier

        float ignore_thresh;
        float iou_thresh;
        float obj_normalizer;
        float max_box_grad;
        float scale_grid;

        float** maxpool_addresses;  // addresses of input layer outputs that were max values (for backprop)
    } layer;

    typedef enum NET_TYPE {
        NET_NONE,
        NET_CLASSIFY,
        NET_DETECT
    } NET_TYPE;

    typedef enum LAYER_TYPE {
        LAYER_INPUT = -1,
        LAYER_NONE,
        LAYER_CONV,
        LAYER_MAXPOOL,
        LAYER_FC,
        LAYER_CLASSIFY,
        LAYER_RESIDUAL,
        LAYER_DETECT,
        LAYER_AVGPOOL,
        LAYER_UPSAMPLE
    } LAYER_TYPE;

    typedef enum LR_POLICY {
        LR_STEPS,
        LR_EXPONENTIAL,
        LR_POLYNOMIAL,
        LR_COSWR
    } LR_POLICY;

    typedef enum ACTIVATION {
        ACT_NONE,
        ACT_RELU,
        ACT_LEAKY,
        ACT_MISH,
        ACT_SIGMOID,
        ACT_SOFTMAX,
        ACT_TANH
    } ACTIVATION;

    typedef enum LOSS_TYPE {
        LOSS_NONE,
        LOSS_MAE,   // mean absolute error
        LOSS_MSE,   // mean squared error
        LOSS_CCE,   // categorical cross-entropy, used for softmax loss
        LOSS_BCE    // binary cross-entropy, used for sigmoid loss
    } LOSS_TYPE;

    typedef enum REGULARIZATION {
        REG_NONE,
        REG_L1,     // sum of absolute values of weights
        REG_L2      // sum of squared values of weights (much preferred over L1)
    } REGULARIZATION;

    typedef struct bbox {
        int lbl;
        float prob;  // probability
        // values are relative (may change idk)
        float cx;
        float cy;
        float w;
        float h;
        float left;
        float right;
        float top;
        float bottom;
        float area;
    } bbox;

    typedef struct det_cell {   // cell of a detection layer used for training
        float top;              // percentage of image width
        float left;
        float bottom;
        float right;
        bbox* tboxes;           // ground truth bounding boxes
        int* obj;               // ground truth objectness in anchors
        int* cls;               // ground truth class id in anchors
    } det_cell;

    typedef struct det_sample { // object detection sample
        size_t n;               // # of bboxes
        bbox* bboxes;
        char* imgpath;
    } det_sample;

    typedef struct class_set {  // classifier class dataset
        char** files;
        size_t class_id;
        size_t n;       // # of files
        size_t* rands;  // array of non-repeating random numbers of size n
        size_t ri;      // rands index
    } class_set;

#ifdef __cplusplus
}
#endif
#endif