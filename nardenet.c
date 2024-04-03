#include "nardenet.h"
#include "xallocs.h"
#include "network.h"
#include "cfg.h"



int main(void) {
	

	char* filename = "D:/TonyDev/NardeNet/nardenet.cfg";
	network* net = create_network_from_cfg(filename);
	print_network(net);
	print_alloc_list();
	
#ifdef _DEBUG
	printf("\n\nPress ENTER to exit the program.");
	(void)getchar();
#endif
}
