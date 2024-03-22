#include "nardenet.h"
#include "xallocs.h"
#include "xopencv.h"


int main() {
#ifdef _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
	printf(" _DEBUG defined.\n");
#endif
	

	//char* filename = "D:/TonyDev/NardeNet/nardenet.cfg";
	//network* net = create_network_from_cfg(filename);
	//print_network(net);
	//image img = load_file_to_image();
	//print_image_matrix(&img);
	//xfree(img.data);
	//free_image(&img);


#ifdef _DEBUG
	printf("\n\nPress ENTER to exit the program.");
	(void)getchar();
#endif
	return 0;
}
