#include <stdio.h>
#include <stdlib.h>
#include "dataset.h"

#ifndef CLUSTERDATA_H
#define CLUSTERDATA_H


void freeClusters();
double* clusterData(Dataset* dataset, int k);

#endif
