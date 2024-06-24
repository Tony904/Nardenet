#include "nardenet.h"
#include "xallocs.h"
#include "network.h"
#include "cfg.h"
#include "train.h"
#include "xopencv.h"
#include "im2col.h"
#include "data.h"
#include "gemm.h"


int main(void) {
	
	TODO: Cost is always zero. Fix.
	char* cfgfile = "D:/TonyDev/NardeNet/cfg/nardenet.cfg";
	network* net = create_network_from_cfg(cfgfile);
	net->n_classes;
	print_network(net);
	net->n_classes;
	
	train(net);

	//free_network(net);
	//print_alloc_list();


#ifndef _DEBUG
	printf("\n\nPress ENTER to exit the program.");
	(void)getchar();
#endif
}
