#include "nardenet.h"
#include "xallocs.h"
#include "network.h"
#include "cfg.h"
#include "train.h"
#include "xopencv.h"
#include "im2col.h"
#include "data.h"
#include "gemm.h"
#include "layer_conv.h"


int main(void) {

	srand(77777774);

	char* cfgfile = "D:/TonyDev/NardeNet/cfg/nardenet.cfg";
	network* net = create_network_from_cfg(cfgfile);
	train(net);

	/*free_network(net);
	print_alloc_list();*/

#ifndef _DEBUG
	printf("\n\nPress ENTER to exit the program.");
	(void)getchar();
#endif
}

/* TODO:

PHASE 1:
 maxpool
 batchnorm
 resize input images if needed
 fully connected tag for cfg files
 learning rate policies: step, adam, adagrad, maybe cosine restart stuff
 saving weights
 loading weights
 other activation functions
 other cost functions
 data augmentation:
	saturation, exposure, hue, mosaic, rotation (classifier only), jitter
 decay (not entirely sure what this is yet)

PHASE 2:
 training progress graph
 upsample layer (not super sure what this is yet)
 spatial pooling (not super sure what this is yet)
 multiple prediction heads
 object detection
	- anchor boxes, IOU, NMS
 GPU support (cuda)

PHASE 3:
 Speed Optimizations
 Python API
 GUI for running inference and training (Python based gui, probably tkinter since it's built in)
 Figure out how to load and display images without OpenCV (to make installation easier)
 Make installation braindead easy

PHASE 4:
 Discover ways to optimize things for manufacturing applications
 Develop an image taking procedure that will provide adequate part detection with minimal images and labeling.
 Orientation detection for both classifier and object detector

PHASE 9999:
 Implement transformer architecture + attention modules
 Mamba/Jamba if that's still a thing by the time I get everything else done
 */