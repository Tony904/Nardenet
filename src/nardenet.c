#include "nardenet.h"
#include "network.h"
#include "cfg.h"
#include "train.h"
#include "image.h"
#include "xopencv.h"
#include "xallocs.h"


int main(void) {

	initialize_xallocs_lock();
	srand(7777777);
	char* cfgfile = "D:\\TonyDev\\Nardenet\\cfg\\nardenet-yolov4-tiny-classifier-tiny-imagenet.cfg";
	network* net = create_network_from_cfg(cfgfile);
	train(net);


	//free_network(net);
	//print_alloc_list();

#ifndef _DEBUG
	printf("\n\nPress ENTER to exit the program.");
	(void)getchar();
#endif
}

/* TODO:
BUGS:

CHANGES:

IN PROGRESS:
 learning rate policies
 setting up yolov4 tiny classifier backbone to train on imagenet and then
	use the weights to train an object detector.

PHASE 1:
 learning rate policies: step, adam, adagrad, rmsprop
 object detection
 data augmentation:
	mosaic, rotation & flip (classifier only), jitter
 swish activation
 ema for last X iterations (ema = exponential moving average)

PHASE 2:
 training progress graph
 GPU support (cuda)
 yolov4-csp implementation
 yolov7 implementation

PHASE 3:
 group norm
 Speed Optimizations
 Python API
 GUI for running inference and training (Python based gui, probably tkinter since it's built in)
 Figure out how to display images without OpenCV (to make installation easier)
 Make installation braindead easy

PHASE 4:
 Discover ways to optimize things for manufacturing applications
 Develop an image taking procedure that will provide adequate part detection with minimal images and labeling.
 Orientation detection for both classifier and object detector

PHASE 9999:
 yolov9 implementation
 Implement transformer architecture + attention modules
 Unsupervised learning
 Adversarial training
 */