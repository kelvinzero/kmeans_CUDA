#include "kmeanscpu.h"

void cpu_calculateCentroidsMeans(double* centroids, int k, double *records, int num_rows, int num_cols){

	int i,j;
	
	// reset centroid vals
	for(i = 0; i < k; i++){
		for(j = 0; j < num_cols; j++)
			centroids[i*num_cols+j] = 0.0;
	}

	// accumulate centroid vals
	for(i = 0; i < num_rows; i++){
		int centroidNum = (int)records[i*num_cols];
		centroids[centroidNum*num_cols] += 1;
		for(j = 1; j < num_cols; j++){
			if(!isnan(records[i*num_cols+j]))
				centroids[centroidNum*num_cols+j] += records[i*num_cols+j];	
		}
	}
	
	// divide sums by counts
	for(i = 0; i < k; i++){
		for(j = 1; j < num_cols; j++){
			if(centroids[i*num_cols] != 0 && centroids[i*num_cols+j] != 0)
				centroids[i*num_cols+j] /= centroids[i*num_cols] ;
			else
				centroids[i*num_cols+j] = 0.0;
		}
	}
}

double cpu_calculateSSE(double *centroids, int k, double * records, int num_rows, int num_cols){
	
	double sse = 0.0;
	int i,j;
	// accumulate sse vals
	for(i = 0; i < num_rows; i++){
		int centroidNum = (int)records[i*num_cols];
		for(j = 1; j < num_cols; j++){
			if(!isnan(records[i*num_cols+j]))
				sse += (records[i*num_cols+j] - centroids[centroidNum*num_cols+j]) *(records[i*num_cols+j] - centroids[centroidNum*num_cols+j]);
		}
	}
	
	return sse;
}

double cpu_euclideanDistance(double *record1, double *record2, int num_cols){

	int i,j;
	double dist = 0.0;
	for(i = 0; i < num_cols; i++){
		dist += (record1[i]-record2[i]) * (record1[i]-record2[i]);
	}
	return sqrt(dist);
} 


void cpu_findClosestCentroids(double* centroids, int k, double *records, int num_rows, int num_cols){

	int closestCluster;
	double closestDistance;
	double currentDistance;

	int i,j;

	for(i = 0; i < num_rows; i++){
		closestCluster = records[i*num_cols];
		closestDistance = cpu_euclideanDistance(&records[i*num_cols], &centroids[closestCluster * num_cols], num_cols);

		for(j = 0; j < k; j++){
			if(j != closestCluster){
				currentDistance = cpu_euclideanDistance(&records[i*num_cols+1], &centroids[j*num_cols+1], num_cols-1);
				if(currentDistance < closestDistance){
					closestDistance = currentDistance;
					closestCluster = j;
				}	
			}
		}
		records[i*num_cols] = closestCluster;
	}
	cpu_calculateCentroidsMeans(centroids, k, records, num_rows, num_cols);
}
		
	
