#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef KMEANSCPU_H
#define KMEANSCPU_H


void cpu_findClosestCentroids(double* d_centroids, int k, double *d_records, int num_rows, int num_cols);
void cpu_calculateCentroidsMeans(double* centroids, int k, double *records, int num_rows, int num_cols);		
double cpu_calculateSSE(double *d_centroids, int k, double * d_records, int num_rows, int num_cols);
double cpu_calculateSSE(double *centroids, int k, double * records, int num_rows, int num_cols);

#endif
