#ifndef NARDENET_H
#define NARDENET_H



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
        STEPS
    } LR_POLICY;

    typedef struct matrix {
        int rows, cols;
        float** vals;
    } matrix;

    typedef struct image {
        int w;
        int h;
        int c;
        float* data;
    } image;

    typedef struct layer {
        LAYER_TYPE type;
        ACTIVATION activation;
        void(*forward)   (struct layer, struct net_state);
        void(*backward)  (struct layer, struct net_state);
        void(*update)    (struct layer, int, float, float, float); //layer, batch, learning rate, momentum, decay
        int batch_size;
        int w;
        int h;
        int c;
        int out_w;
        int out_h;
        int out_c;
        int n_filters;
        int filter_size;
        int stride;
        int padding;
        int n_inputs;
        int n_outputs;
        float* output;
        int n_weights; // = c * n_filter * filter_size * filter_size;
        float* weights;
        float* biases;
        float* delta;
        float* means;
        float* variances;
        float* losses;
        int train;

    } layer;

    typedef struct network {
        int n_layers;
        int w;
        int h; 
        int c;
        int batch_size;  // number of images per batch
        int subbatch_size;  // number of images per sub-batch, must divide evenly into batch_size
        int max_iterations;  // maximum number of training iterations before automatically ending training
        float* learning_rate;
        float momentum;
        float decay;
        float* output;
        int iteration;  // current iteration, one iteration = one batch processed
        int epoch_size;  // number of batches to complete one epoch
        float saturation[2];
        float exposure[2];
        float hue[2];
        LR_POLICY policy;
        layer* layers;

    } network;

    typedef struct net_state {
        float* truth;
        float* input;
        float* delta;
        float* workspace;
        int train;
        int index;
    } net_state;

#ifdef __cplusplus
}
#endif
#endif
