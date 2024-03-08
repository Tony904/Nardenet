#include "nardenet.h"
#include <stdio.h>
#include <stdlib.h>

#include "xallocs.h"
#include "network.h"
#include "cfg_reader.h"






int main() {

	int num_of_layers = 3;
	network net = make_network(num_of_layers);
	printf("Learning Rate: %f\n", net.learning_rate[0]);
	


#ifdef _DEBUG
	printf("\n\nPress ENTER to exit the program.");
	getchar();
#endif
	return 0;
}
