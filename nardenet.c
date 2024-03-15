#include "nardenet.h"
#include <stdio.h>
#include <stdlib.h>

#include "xallocs.h"
#include "network.h"
#include "config.h"






int main() {

	int num_of_layers = 3;
	network net = new_network(num_of_layers);
	char* filename = "D:/TonyDev/NardeNet/test.cfg";
	load_cfg(filename, &net);

	//list lst = *make_list();
	//char* string = "hello";
	//list_append(&lst, string);
	//printf("list size: %zu\n", lst.size);
	//printf("first item: %s", (char*)lst.first->val);


#ifdef _DEBUG
	printf("\n\nPress ENTER to exit the program.");
	(void)getchar();
#endif
	return 0;
}
