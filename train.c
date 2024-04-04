#include "train.h"

#include <stdio.h>
#include <math.h>


void initialize_weights_biases_random(network* net);
double randn(double mean, double stddev);


void train(network* net) {
	initialize_weights_biases_random(net);
}

void initialize_weights_biases_random(network* net) {
	net;
	srand(7777777);
	float x;
	size_t n = 100;
	double stddev = sqrt(2.0 / 3 * 3 * 28);
	for (size_t i = 0; i < n; i++) {
		x = (float)randn(0.0, stddev);
		printf("%f\n", x);
	}
}

double randn(double mean, double stddev) {
	static double n2 = 0.0;
	static int n2_cached = 0;
	if (n2_cached) {
		n2_cached = 0;
		return n2 * stddev + mean;
	}
	else {
		double x = 0.0;
		double y = 0.0;
		double r = 0.0;
		while (r == 0.0 || r > 1.0) {
			x = 2.0 * rand() / RAND_MAX - 1;
			y = 2.0 * rand() / RAND_MAX - 1;
			r = x * x + y * y;
		} 
		double d = sqrt(-2.0 * log(r) / r);
		n2 = y * d;
		n2_cached = 1;
		return x * d * stddev + mean;
	}
	
}



