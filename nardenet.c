#include "nardenet.h"
#include "xallocs.h"
#include "network.h"
#include "cfg.h"
#include "train.h"
#include "xopencv.h"
#include "im2col.h"



int main(void) {
	

	char* cfgfile = "D:/TonyDev/NardeNet/nardenet.cfg";
	network* net = create_network_from_cfg(cfgfile);
	print_network(net);
	
	image* img = load_file_to_image("D:\\TonyDev\\NardeNet\\images\\one_28x28.jpg");
	net->input = img;
	train(net);

	layer* l = &net->layers[net->n_layers - 1];
	image dst = { 0 };
	dst.data = l->output;
	dst.w = l->out_w;
	dst.h = l->out_h;
	dst.c = 3;
	print_image_matrix(&dst);
	show_image(&dst);

	//free_network(net);
	//print_alloc_list();

	//test_im2col();


#ifdef _DEBUG
	printf("\n\nPress ENTER to exit the program.");
	(void)getchar();
#endif
}
