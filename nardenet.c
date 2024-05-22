#include "nardenet.h"
#include "xallocs.h"
#include "network.h"
#include "cfg.h"
#include "train.h"
#include "xopencv.h"
#include "im2col.h"
#include "data.h"


int main(void) {
	

	char* datafile = "D:/TonyDev/NardeNet/paths.data";
	data_paths* dp = get_data_paths(datafile);
	network* net = create_network(dp);
	//print_network(net);
	
	train(net);

	/*layer* l = &net->layers[net->n_layers - 1];
	image dst = { 0 };
	dst.data = l->output;
	dst.w = l->out_w;
	dst.h = l->out_h;
	dst.c = 3;
	print_image_matrix(&dst);
	show_image(&dst);*/

	//free_network(net);
	//print_alloc_list();

	//test_im2col();


#ifdef _DEBUG
	printf("\n\nPress ENTER to exit the program.");
	(void)getchar();
#endif
}
