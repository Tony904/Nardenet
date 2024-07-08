#include "nardenet.h"
#include "network.h"
#include "cfg.h"
#include "train.h"
#include "layer_maxpool.h"
#include "image.h"


int main(void) {

	srand(7777777);

	/*char* cfgfile = "D:/TonyDev/NardeNet/cfg/nardenet.cfg";
	network* net = create_network_from_cfg(cfgfile);
	train(net);*/

	write_image();
	image* img = load_image("C:\\Users\\TNard\\OneDrive\\Desktop\\dev\\Nardenet-main\\data\\testimg.bmp");
	pprint_mat(img->data, (int)img->w, (int)img->h, (int)img->c);

	/*image* img = load_image("D:\\TonyDev\\NardeNet\\data\\cifar3\\train\\bird\\bird1.png");
	show_image(img);*/

	/*free_network(net);
	print_alloc_list();*/

#ifndef _DEBUG
	printf("\n\nPress ENTER to exit the program.");
	(void)getchar();
#endif
}

/* TODO:
BUGS:

PHASE 1:
 batchnorm
	- forward pass done, todo backward pass and test time variation
 resize input images if needed
 fully connected tag for cfg files
 learning rate policies: step, adam, adagrad, rmsprop, cyclic rates maybe
 saving weights
 loading weights
 other activation functions
 other loss functions
 data augmentation:
	saturation, exposure, hue, mosaic, rotation (classifier only), jitter
 decay (not entirely sure what this is yet)

PHASE 2:
 training progress graph
 group convolution (i.e. at group = 2, half of filters convolve over one half of the channels, the other
	half of filters convolve over the other half of the filters. note: sometimes group convolutions
	are followed up by a 1x1 convolution to maintain the same feature pooling as a normal 3x3 convolution.
	video on group convs: https://www.youtube.com/watch?v=vVaRhZXovbw)
 upsample layer (not super sure what this is yet)
 spatial pooling (not super sure what this is yet)
 multiple prediction heads
 object detection
	- anchor boxes, IOU, NMS
 GPU support (cuda)

PHASE 3:
 group norm
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
 Unsupervised learning
 Adversarial training
 */