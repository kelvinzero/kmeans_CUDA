#include <stdio.h>
#include <stdlib.h>

#ifndef SSEKERNEL_H
#define SSEKERNEL_H


__global__ void calculateSSE(double* centroids, size_t centroidsPitch, double* records, size_t recordsPitch, int rows, int cols, double *SSE);
__global__ void findClosestClusters(double* centroids, size_t centroidsPitch, int k, double* records, size_t recordsPitch, int rows, int cols);
__global__ void calculateCentroidMeans(double* centroid_results, size_t centroidsPitch, int k, double* records, size_t recordspitch, int rows, int cols);
#endif
