#include "clusterdata.h"
#include "timing.h"

int SHAREDATA_ROWS = 64;
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
	dim3 griddimSSE;
	dim3 blockdimReduce;

	size_t centroidsPitch;
	size_t resultsPitch;
	size_t recordsPitch;

	size_t sharedsize;
	size_t clustersize;
	size_t recordsize; 
	size_t sharedSSEsize;

	double *d_centroids;
	double *d_records;
	double *d_results;
	double *h_results;
	double *d_SSE;
	double *h_SSE;
	double lastSSE;
	double currentSSE;
	double sseTime;
	double totalTime;
	float  time = 0;
	int i,j;

   	cudaEvent_t launch_begin, launch_end, sse_begin, sse_end;
   	cudaEventCreate(&launch_begin);
   	cudaEventCreate(&launch_end);
   	cudaEventCreate(&sse_begin);
   	cudaEventCreate(&sse_end);

	blockdim.x 	= 1;
	blockdim.y 	= 256;
	blockdimSSE.x 	= num_cols-1;
	blockdimSSE.y 	= (int)(256/blockdimSSE.x);
	griddimSSE.x 	= 1;
	griddimSSE.y 	= ceil((float)num_rows/blockdimSSE.y);
	griddim.x 	= 1;	
	griddim.y 	= ceil((float)num_rows/256);
	blockdimReduce.x = num_cols;
	blockdimReduce.y = k;

	h_SSE		= (double*)calloc(griddimSSE.y, sizeof(double));	
	sharedsize 	= SHAREDATA_ROWS * num_cols * sizeof(double);	
	clustersize 	= k*num_cols*sizeof(double);
	recordsize 	= num_rows*num_cols*sizeof(double);	
	sharedSSEsize 	= blockdimSSE.x * blockdimSSE.y * sizeof(double);
	h_results	= (double*)malloc(griddim.y*k*num_cols*sizeof(double));

	printf("\n");
	printf("Find clusters: BlockDim.x: %d BlockDim.y: %d GridDim.x %d GridDim.y %d\n", blockdim.x, blockdim.y, griddim.x, griddim.y);
	printf("Calculate SSE: BlockDim.x: %d BlockDim.y: %d GridDim.x %d GridDim.y %d\n", blockdimSSE.x, blockdimSSE.y, 1, 1);
	printf("\nAllocating/memcpy device memory\n");

	cudaMallocPitch((void**)&d_centroids, &centroidsPitch, num_cols*sizeof(double), k);
	cudaMallocPitch((void**)&d_records, &recordsPitch, num_cols*sizeof(double), num_rows);
	cudaMallocPitch((void**)&d_results, &resultsPitch, num_cols*sizeof(double), griddim.y*k);
	cudaMalloc((void**)&d_SSE, griddimSSE.y*sizeof(double));

	cudaMemcpy2D(d_centroids, centroidsPitch, centroids, num_cols*sizeof(double), num_cols*sizeof(double), k, cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_records, recordsPitch, records, num_cols*sizeof(double), num_cols*sizeof(double), num_rows, cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_results, resultsPitch, h_results, num_cols*sizeof(double), num_cols*sizeof(double), griddim.y*k, cudaMemcpyHostToDevice);
	
	printf("\nClustering..\n");	
	
	i = 0;
	currentSSE = 0;
	lastSSE = 1;
	totalTime = 0;
	sseTime = 0;
	while(i < MAXRUNS && lastSSE > currentSSE){
		i++;
		time = 0;
		lastSSE = currentSSE;
		
		cudaEventRecord(launch_begin,0);
		findClosestClusters<<<griddim,blockdim, sharedsize>>>(d_centroids, centroidsPitch/sizeof(double), k, d_records, recordsPitch/sizeof(double),  num_rows, num_cols);
		cudaDeviceSynchronize();
		calculateCentroidMeans<<<1, blockdimReduce, clustersize>>>(d_centroids, centroidsPitch/sizeof(double), k, d_records, recordsPitch/sizeof(double), num_rows, num_cols); 
		cudaDeviceSynchronize();
		cudaEventRecord(launch_end,0);
		cudaEventSynchronize(launch_end);
    		cudaEventElapsedTime(&time, launch_begin, launch_end);

		totalTime += (time/1000);
		time = 0;

		cudaEventRecord(sse_begin,0);		
		calculateSSE<<<griddimSSE, blockdimSSE, sharedSSEsize>>>(d_centroids, centroidsPitch/sizeof(double), d_records, recordsPitch/sizeof(double), num_rows, num_cols, d_SSE); 
		cudaDeviceSynchronize();
		cudaEventRecord(sse_end,0);
		cudaEventSynchronize(sse_end);		
  		cudaEventElapsedTime(&time, sse_begin, sse_end);
		sseTime += (time/1000);
		
		cudaMemcpy(h_SSE, d_SSE, griddimSSE.y * sizeof(double),  cudaMemcpyDeviceToHost);
		currentSSE=0;	
		for(j = 0; j < griddimSSE.y; j++){
			currentSSE += h_SSE[j];
		}

		if(i == 1)
			lastSSE = currentSSE+1;
		
		if(lastSSE > currentSSE){
			cudaMemcpy(records, d_records, recordsize, cudaMemcpyDeviceToHost);
			cudaMemcpy(centroids, d_centroids, clustersize, cudaMemcpyDeviceToHost);
		}
		
		time = 0;
	}

	cudaMemcpy2D(records, num_cols*sizeof(double), d_records, recordsPitch, num_cols*sizeof(double), num_rows, cudaMemcpyDeviceToHost);
	cudaMemcpy2D(h_results, num_cols*sizeof(double), d_results, resultsPitch, num_cols*sizeof(double), k*griddim.y, cudaMemcpyDeviceToHost);
	cudaMemcpy2D(centroids, num_cols*sizeof(double), d_centroids, centroidsPitch, num_cols*sizeof(double), k, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	printf("\nClustering complete in %d iterations\n", i-1);
	printf("Completed in %lf seconds\n", totalTime + sseTime);
	printf("Average run time: %lf\n", (totalTime+sseTime)/(i-1));
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

void beginCpuClustering(double *centroids, double* records, int k, int num_rows, int num_cols){

    	float clustertime, ssetime;
    	clock_t now, then;
	clock_t nowsse, thensse;

	double sse = 0;
	double lastsse = 1;
	int i = 0;

	while(lastsse > sse && i++ < MAXRUNS) {
		lastsse = sse;
		then = clock();
		cpu_findClosestCentroids(centroids, k, records, num_rows, num_cols);
		now = clock();
		thensse = clock();
		sse = cpu_calculateSSE(centroids, k, records, num_rows, num_cols);
		nowsse = clock();
		
		clustertime += timeCost(then, now);
		ssetime += timeCost(thensse, nowsse);

		if(i == 1)
			lastsse = sse+1;
	}
	printf("\nClustering complete in %d iterations\n", i-1);
	printf("Completed in %lf seconds\n", clustertime + ssetime);
	printf("Average run time: %lf\n", (clustertime+ssetime)/(i-1));
	printf("Total reclustering time: %lf\n", clustertime);
	printf("Average time to recluster: %lf\n", clustertime/(i-1));
	printf("Total SSE calculating time: %lf\n", ssetime);
	printf("Average time to calculate SSE: %lf\n", ssetime/(i-1));
}

/*
/ clusterData()		Converts the dataset to numeric values and centroids
/					Clusters the records in the file into k centroids (gpu)
/					Uses squared error check for convergence (gpu)
*/
double* clusterData(Dataset* dataset, double **centroids, int k, int gpu){

	*centroids = (double*)calloc(k * (dataset->num_cols+1), sizeof(double));
	double* numeric_records = loadDatasetNumeric(dataset);
	setFirstClusters(*centroids, numeric_records, k, dataset->num_rows-1, dataset->num_cols+1);	
	if(gpu)
		beginGpuClustering(*centroids, numeric_records, k, dataset->num_rows-1, dataset->num_cols+1);	
	else
		beginCpuClustering(*centroids, numeric_records, k, dataset->num_rows-1, dataset->num_cols+1);

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

	printf("Assigning random records to centroids..\n");

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
			if(j==0)
				centroids[i*cols] = 0;
			else
				centroids[i*cols+j] = records[randn * cols + j];
		}
	}
	free(rands);
}
