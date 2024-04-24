#include "nardenet.h"
#include "xallocs.h"
#include "network.h"
#include "cfg.h"
#include "train.h"
#include "xopencv.h"
#include "im2col.h"



int main(void) {
	

	/*char* filename = "D:/TonyDev/NardeNet/nardenet.cfg";
	network* net = create_network_from_cfg(filename);
	print_network(net);*/
	
	//train(net);

	//free_network(net);
	//print_alloc_list();

	test_img2col();




#ifdef _DEBUG
	printf("\n\nPress ENTER to exit the program.");
	(void)getchar();
#endif
}
