#include <stdio.h>
#include <stdlib.h>
#include "dataset.h"
#include "kmeanscpu.h"
#include "clusterkernel.h"


#ifndef CLUSTERDATA_H
#define CLUSTERDATA_H


void freeClusters();
double* clusterData(Dataset* dataset, double **centroids, int k, int gpu);

#endif
