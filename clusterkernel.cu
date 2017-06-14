#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>

#include "clusterkernel.h"

__device__ inline void atomicAddFloat(float* address, float value){

      float old = value;  
      float new_old;

      do {
    	new_old = atomicExch(address, 0.0f);
    	new_old += old;
      }
      while ((old = atomicExch(address, new_old))!=0.0f);
};

/**
/ euclideanDistance()		Calculates euclidean distance between two records
*/
__device__ double euclideanDistance(double *record1, double *record2, int cols){
	
	double dist = 0.0f;
	int i;
	for(i = 1; i < cols; i++){
		dist += (record1[i]-record2[i]) * (record1[i]-record2[i]);
	}
	return sqrt(dist);
}

/**
/ calculateCentroidMeans()	Calculates mean values for each centroid element
/	
/							Each thread sums a single attribute for each record 
/								belonging to the cluster for that thread.
*/
__global__ void calculateCentroidMeans(double* centroids, size_t centroidsPitch, int k, double* records, size_t recordsPitch, int rows, int cols){

	extern __shared__ double s_centroids[];
	int i;

	s_centroids[threadIdx.y*cols+threadIdx.x] = 0.0;

	for(i = 0; i < rows; i++){
		int currentCentroid = records[i*recordsPitch];
		if(currentCentroid == threadIdx.y){
			if(threadIdx.x > 0)
				s_centroids[threadIdx.y*cols+threadIdx.x] += records[i*recordsPitch+threadIdx.x];
			
			else
				s_centroids[threadIdx.y*cols] += 1;
		}
	}
	__syncthreads();
	
	if(threadIdx.x == 0){
		centroids[threadIdx.y*centroidsPitch] = s_centroids[threadIdx.y*cols];
		return;
	}
	__syncthreads();

	if(s_centroids[threadIdx.y*cols] == 0 || isnan(s_centroids[threadIdx.y*cols]) || isnan(s_centroids[threadIdx.y*cols+threadIdx.x]) || s_centroids[threadIdx.y*cols+threadIdx.x]==0)	
		centroids[threadIdx.y*centroidsPitch+threadIdx.x] = 0.0;
	else
		centroids[threadIdx.y*centroidsPitch+threadIdx.x] = s_centroids[threadIdx.y*cols+threadIdx.x]/s_centroids[threadIdx.y*cols]; 
}

/**
/ findClosestClusters()		Record blocks assign each record to closest cluster 
*/
__global__ void findClosestClusters(double* centroids, size_t centroidsPitch, int k, double* records, size_t recordsPitch, int rows, int cols){

	extern __shared__ double s_centroids[];

	int idx = 0;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int sharedIdx = threadIdx.y*cols;
	int recordIdx = idy*recordsPitch;
	
	if(idx >= cols || idy >= rows)
		return;
	
		
		int closestCluster = 0;
		double closestDistance = euclideanDistance(&records[sharedIdx], &centroids[closestCluster * centroidsPitch], cols);
		double thisDistance;
		int i;
		for(i = 0; i < k; i++){
			thisDistance = euclideanDistance(&records[recordIdx], &centroids[i*centroidsPitch], cols);
			double diff = thisDistance - closestDistance;
			if(diff < 0){
				closestDistance = thisDistance;
				closestCluster = i;
			}
		
		}
		records[idy * recordsPitch] = closestCluster;
}



/**
/ calculateSSE()	Calculates the sum of squared errors of all records to assigned centroid
*/
__global__ void calculateSSE(double* centroids, size_t centroidsPitch, double* records, size_t recordsPitch, int rows, int cols, double *SSE){

	
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	extern __shared__ double s_SSE[];
	
	s_SSE[threadIdx.y*blockDim.x+threadIdx.x] = 0;

	
	if(idy >= rows || threadIdx.x >= cols)
		return;


	int myCluster = records[idy * recordsPitch];
	double myval = records[idy * recordsPitch + threadIdx.x+1];
	double avg = centroids[myCluster*centroidsPitch+threadIdx.x+1];
	if(isnan(myval))
		s_SSE[threadIdx.y*blockDim.x+threadIdx.x] = -avg;
	else
		s_SSE[threadIdx.y*blockDim.x+threadIdx.x] = (myval-avg) * (myval-avg);
	__syncthreads();

	if(threadIdx.x==0){
		int i;
		for(i = 1; i< blockDim.x; i++)
			s_SSE[threadIdx.y*blockDim.x] += s_SSE[threadIdx.y*blockDim.x+i];
	}
	__syncthreads();
	if(threadIdx.x == 0 && threadIdx.y == 0){
		int i;
		for(i = 0; i < blockDim.y; i++)
			s_SSE[0] += s_SSE[i*blockDim.x];
		SSE[blockIdx.y] = s_SSE[0];
	}
}

