#include "kmeanskernel.h"

__device__ double euclideanDistance(double *record1, double *record2, int cols){
	
	double dist = 0.0f;
	int i;
	for(i = 1; i < cols; i++){
		dist += (record1[i]-record2[i]) * (record1[i]-record2[i]);
	}
	return sqrt(dist);
}

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

	if(threadIdx.x > 0)
		centroids[centroidNum*cols+threadIdx.x] = centroidAttribute / centroids[centroidNum*cols];

}


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
		if(threadIdx.y == 0){
			dim3 blockdim;
			blockdim.x = cols;
			blockdim.y = k;
		        calculateCentroidMeans<<<1, blockdim>>>(centroids, k, records, rows, cols);
		}
	}
}
