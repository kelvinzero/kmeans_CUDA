#include "kmeanskernel.h"

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
__global__ void calculateCentroidMeans(double* centroids, int k, double* records, int rows, int cols){

	int i;
	int count = 0;
	int centroidNum = threadIdx.y;
	double centroidAttribute = 0.0;

	if(threadIdx.x >= cols || threadIdx.y >= k)
		return;

	for(i = 0; i < rows; i++){
		int recordCentroidNum = (int)records[i*cols];
		if(recordCentroidNum == centroidNum){
			if(threadIdx.x == 0)
				count++;	
			else{
				double recordVal = records[i*cols+threadIdx.x];
				if(!isnan(recordVal)){
					centroidAttribute += recordVal;
				}
			}
		}
	}
	
	if(threadIdx.x == 0)
		centroids[centroidNum*cols] = count;

	__syncthreads();

	if(threadIdx.x > 0){
		if(centroids[centroidNum*cols] > 0)
			centroids[centroidNum*cols+threadIdx.x] = centroidAttribute / centroids[centroidNum*cols];
		else
			centroids[centroidNum*cols+threadIdx.x] = 0.0;
	}
}

/**
/ findClosestClusters()		Record blocks assign each record to closest cluster 
*/
__global__ void findClosestClusters(double* centroids, int k, double* records, int rows, int cols){

	extern __shared__ double s_records[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int sharedIdx = threadIdx.y*cols + threadIdx.x;
	int recordIdx = idy*cols + threadIdx.x;
	if(idx >= cols || idy >= rows)
		return;
	s_records[sharedIdx] = records[recordIdx];
	__syncthreads();
	
	if(threadIdx.x > 0)
		return;	

	if(threadIdx.x==0){
		int closestCluster = (int)s_records[sharedIdx];
		double closestDistance = euclideanDistance(&s_records[sharedIdx], &centroids[closestCluster * cols], cols);
		double thisDistance;
		int i;
		for(i = 0; i < k; i++){
			thisDistance = euclideanDistance(&s_records[sharedIdx], &centroids[i * cols], cols);
			double diff = thisDistance - closestDistance;
			if(diff < 0){
				closestDistance = thisDistance;
				s_records[sharedIdx] = i;
			}
		
		}
		records[idy*cols] = s_records[sharedIdx];	
		__syncthreads();
		if(threadIdx.y == 0){
			dim3 blockdim;
			blockdim.x = cols;
			blockdim.y = k;
		        calculateCentroidMeans<<<1, blockdim>>>(centroids, k, records, rows, cols);
		}
	}
}

/**
/ calculateSSE()	Calculates the sum of squared errors of all records to assigned centroid
*/
__global__ void calculateSSE(double* centroids, int k, double* records, int rows, int cols, double *SSE){

	extern __shared__ double s_SSE[];

	int	i;
	int 	recordCluster;
	double 	centroidAttributeMean;
	double 	recordVal;
	double 	SSEtotal;	
	double 	xMinusXm;
	double 	thisSSE 	= 0.0;
	int	idx 		= threadIdx.x;
	int 	idy 		= threadIdx.y;
	int 	SSEindex 	= idy * (cols-1) + idx;
	int     centroidIndex   = idy * cols + idx + 1;
	
	if(idx >= cols-1 || idy >= k)
		return;
	
	centroidAttributeMean = centroids[centroidIndex];
	
	// check if each record's centroid index belongs to this centroid
	for(i = 0; i < rows; i++){

		recordCluster = records[i * cols]; 
		if(recordCluster == idy){ // if belongs to this centroid

			recordVal = records[i*cols+idx+1];
			if(!isnan(recordVal)){
				xMinusXm = recordVal - centroidAttributeMean;
				thisSSE += (xMinusXm * xMinusXm);
			}
		}
	}	
	
	
	// set the s_SSE for this centroid attribute
	s_SSE[SSEindex] = thisSSE;	
	__syncthreads();

	// add all SSE values for this SSE row
	if(idx==0){
		
		for(i = 1; i < cols-1; i++){
			thisSSE += s_SSE[SSEindex+i];
		}
		// store sum of squared errors to this centroids shared SSE, column = 0
		s_SSE[SSEindex] = thisSSE; 
	}

	__syncthreads();

	// sum all the SSE in column 0 and return the total
	if(idx==0 && idy == 0){
		SSEtotal = 0.0;
		for(i = 0; i < k; i++)
			SSEtotal += s_SSE[i*cols];
		
		SSE[0] = SSEtotal;
	}
}
