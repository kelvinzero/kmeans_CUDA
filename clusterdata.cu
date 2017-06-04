#include "clusterdata.h"
#include "kmeanskernel.h"

int SHAREDATA_ROWS = 16;
int MAXRUNS = 50;

double* loadDatasetNumeric(Dataset* dataset);
void setFirstClusters(double* centroids, double* records, int k, int rows, int cols);


/*
/ beginGpuCluster()	Sets up kernel and performs clustering
/			Records find cluster on GPU, SSE performed on GPU	
*/
double beginGpuClustering(double* centroids, double *records, int k, int num_rows, int num_cols){

	cudaSetDevice(2);
	cudaDeviceReset();

	dim3 blockdim;
	dim3 griddim;
	dim3 blockdimSSE;

	size_t sharedsize;
	size_t clustersize;
	size_t recordsize; 
	size_t sharedSSEsize;

	double *d_centroids;
	double *d_records;
	double *d_SSE;
	double *h_SSE;
	double lastSSE;
	double currentSSE;
	double sseTime;
	double totalTime;
	float  time = 0;
	int i;

   	cudaEvent_t launch_begin, launch_end, sse_begin, sse_end;
   	cudaEventCreate(&launch_begin);
   	cudaEventCreate(&launch_end);
   	cudaEventCreate(&sse_begin);
   	cudaEventCreate(&sse_end);

	blockdim.x 	= num_cols;
	blockdim.y 	= SHAREDATA_ROWS;
	blockdimSSE.x 	= num_cols-1;
	blockdimSSE.y 	= k;
	griddim.x 	= 1;
	griddim.y 	=  ceil((float)num_rows/SHAREDATA_ROWS);
	h_SSE		= (double*)calloc(1, sizeof(double));	
	sharedsize 	= SHAREDATA_ROWS * num_cols * sizeof(double);	
	clustersize 	= k*num_cols*sizeof(double);
	recordsize 	= num_rows*num_cols*sizeof(double);	
	sharedSSEsize 	= k*(num_cols-1)*sizeof(double);

	printf("\n\n");
	printf("Find clusters: BlockDim.x: %d BlockDim.y: %d GridDim.x %d GridDim.y %d\n", blockdim.x, blockdim.y, griddim.x, griddim.y);
	printf("Calculate SSE: BlockDim.x: %d BlockDim.y: %d GridDim.x %d GridDim.y %d\n", blockdimSSE.x, blockdimSSE.y, 1, 1);
	printf("\nAllocating/memcpy device memory\n");
	cudaMalloc((void**)&d_centroids, clustersize);
	cudaMalloc((void**)&d_records, recordsize);
	cudaMalloc((void**)&d_SSE, sizeof(double));
	cudaMemcpy(d_centroids, centroids, clustersize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_records, records, recordsize, cudaMemcpyHostToDevice);
	
	printf("\nClustering..\n");	
	
	i = 0;
	currentSSE = 0;
	lastSSE = 1;
	totalTime = 0;
	sseTime = 0;
	while(i++ < MAXRUNS && lastSSE > currentSSE){

		time = 0;
		lastSSE = currentSSE;
		
		cudaEventRecord(launch_begin,0);
		findClosestClusters<<<griddim,blockdim, sharedsize>>>(d_centroids, k, d_records, num_rows, num_cols);
		cudaDeviceSynchronize();

		cudaEventRecord(launch_end,0);
		cudaEventSynchronize(launch_end);
    		cudaEventElapsedTime(&time, launch_begin, launch_end);

		totalTime += (time/1000);
		time = 0;

		cudaEventRecord(sse_begin,0);		
		calculateSSE<<<1, blockdimSSE, sharedSSEsize>>>(d_centroids, k, d_records, num_rows, num_cols, d_SSE);
		cudaEventRecord(sse_end,0);
		cudaEventSynchronize(sse_end);		
    		cudaEventElapsedTime(&time, sse_begin, sse_end);
		sseTime += (time/1000);
		
		cudaMemcpy(h_SSE, d_SSE, sizeof(double), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		currentSSE = h_SSE[0];

		if(i == 1)
			lastSSE = currentSSE+1;

		if(lastSSE > currentSSE){
			cudaMemcpy(records, d_records, recordsize, cudaMemcpyDeviceToHost);
			cudaMemcpy(centroids, d_centroids, clustersize, cudaMemcpyDeviceToHost);
		}
		
		time = 0;
    		// measure the time spent in the kernel
	}
	printf("\nClustering complete in %d iterations\n", i-1);
	printf("Completed in %lf seconds\n", totalTime + sseTime);
	printf("Total reclustering time: %lf\n", totalTime);
	printf("Average time to recluster: %lf\n", totalTime/(i-1));
	printf("Total SSE calculating time: %lf\n", sseTime);
	printf("Average time to calculate SSE: %lf\n", sseTime/(i-1));
	free(h_SSE);
	cudaEventDestroy(launch_begin);
	cudaEventDestroy(launch_end);
	cudaEventDestroy(sse_begin);
	cudaEventDestroy(sse_end);
	cudaFree(d_centroids);
	cudaFree(d_records);
	cudaFree(d_SSE);

	return currentSSE;
}

/*
/ clusterData()		Converts the dataset to numeric values and centroids
/					Clusters the records in the file into k centroids (gpu)
/					Uses squared error check for convergence (gpu)
*/
double* clusterData(Dataset* dataset, double **centroids, int k){

	*centroids = (double*)calloc(k * (dataset->num_cols+1), sizeof(double));
	double* numeric_records = loadDatasetNumeric(dataset);



	setFirstClusters(*centroids, numeric_records, k, dataset->num_rows-1, dataset->num_cols+1);	



	beginGpuClustering(*centroids, numeric_records, k, dataset->num_rows-1, dataset->num_cols+1);	

	return numeric_records;
}

/**
/ loadDatasetNumeric()	Converts the set of strings to double values
/			Loads a flattened double* array with record values
/			Numeric records and cluster records have an additional column (col+1)
*/
double* loadDatasetNumeric(Dataset* dataset){

	int thisrow, thiscol;
	int num_rows = dataset->num_rows-1;
	int num_cols = dataset->num_cols+1;
	char** records = dataset->records;
	double *DATASET_NUMERIC;
	
	DATASET_NUMERIC = (double*)calloc(num_rows * num_cols, sizeof(double));

	for(thisrow = 0; thisrow < num_rows; thisrow++){

		for(thiscol = 0; thiscol < num_cols; thiscol++){
			int numericIdx = thisrow*num_cols+thiscol;
			int recordIdx = (thisrow+1)*(num_cols-1)+thiscol-1;
			if(thiscol == 0)
				DATASET_NUMERIC[numericIdx] = 0;
			else{
				sscanf(records[recordIdx], "%lf", (double*)&DATASET_NUMERIC[numericIdx]);
			}
		}
	}
	return DATASET_NUMERIC;

}

/**
/ setFirstClusters() 	Sets the initial cluster numbers and records
/			Assigned a random record value to each cluster
*/
void setFirstClusters(double* centroids, double* records, int k, int rows, int cols){

	int i,j, r;
	time_t t;
	int *rands = (int*)calloc(k, sizeof(int));
	srand((unsigned) time(&t));
	int randn;

	printf("***\n\tAssigning random records to centroids..\n\n");

	// set initial cluster numbers
	for(i = 0; i < k; i++){
	
	randn = abs(rand() % (rows-1));
		int done = false;
		
		while(!done){
			done = true; 
			for(r = 0; r < k; r++){
				if(randn == rands[r]){
					done = false;
					randn = abs(rand() % (rows-1));
		}}}
		
		rands[i] = randn;

		for(j = 0; j < cols; j++){
			if(j==0){
				centroids[i*cols] = 0;
				printf("Assigning - Centroid: [%d] Record: [%d]\n", i, randn);
			}
			else
				centroids[i*cols+j] = records[randn * cols + j];
		}
	}
	free(rands);
}
