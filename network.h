#ifndef NETWORK_H
#define NETWORK_H

#include "nardenet.h"
#include "xallocs.h"

#ifdef __cplusplus
extern "C" {
#endif


network* new_network(int num_of_layers);
void free_network(network net);


#ifdef __cplusplus
}
#endif
#endif