#include <stdio.h>
#include <stdlib.h>

#ifndef KMEANSKERNEL_H
#define KMEANSKERNEL_H


__global__ void findClosestClusters(double* centroids, int k, double* records, int rows, int cols);
__global__ void calculateSSE(double* centroids, int k, double* records, int rows, int cols, double *SSE);
__device__ double euclideanDistance(double *record1, double *record2, int cols);

#endif
