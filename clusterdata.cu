#include "clusterdata.h"
#include "kmeanskernel.h"

int SHAREDATA_ROWS = 16;

double* loadDatasetNumeric(Dataset* dataset);
void setFirstClusters(double* centroids, double* records, int k, int rows, int cols);


/*
/ beginGpuCluster()	Sets up kernel and performs clustering
/			Records find cluster on GPU, SSE performed on GPU	
*/
double* beginGpuClustering(double* centroids, double *records, int k, int num_rows, int num_cols){

	cudaSetDevice(3);
	cudaDeviceReset();
	dim3 blockdim;
	dim3 griddim;
	size_t sharedsize;
	size_t clustersize;
	size_t recordsize; 

	blockdim.x = num_cols;
	blockdim.y = SHAREDATA_ROWS;
	griddim.x = 1;
	griddim.y =  ceil((float)num_rows/SHAREDATA_ROWS);
	sharedsize = SHAREDATA_ROWS * num_cols * sizeof(double);	
	clustersize = k*num_cols*sizeof(double);
	recordsize = num_rows*num_cols*sizeof(double);	

	double *d_centroids;
	double *d_records;
	
	int i,j;
	printf("\n\n");
	printf("BlockDim.x: %d BlockDim.y: %d GridDim.x %d GridDim.y %d\n", blockdim.x, blockdim.y, griddim.x, griddim.y);
	cudaMalloc((void**)&d_centroids, clustersize);
	cudaMalloc((void**)&d_records, recordsize);
	cudaMemcpy(d_centroids, centroids, clustersize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_records, records, recordsize, cudaMemcpyHostToDevice);
	
	findClosestClusters<<<griddim,blockdim, sharedsize>>>(d_centroids, k, d_records, num_rows, num_cols);
	
	cudaMemcpy(centroids, d_centroids, clustersize, cudaMemcpyDeviceToHost);
	cudaMemcpy(records, d_records, recordsize, cudaMemcpyDeviceToHost);

	
	printf("\n***\n\tCentroids:\n\n");
	for(i = 0; i < k; i++){
		printf("Centroid [%d]: ", i);
		for(j = 0; j < num_cols; j++){
			printf("%f ", centroids[i*num_cols+j]);
		}
		printf("\n");
	}
	printf("\nRecords:\n\n");
	for(i = 0; i < 20; i++){
		for(j = 0; j < num_cols; j++){
			printf("%f ", records[i*num_cols+j]);
		}
		printf("\n");
	}
	return NULL;
}

/*
/ clusterData()		Converts the dataset to numeric values and centroids
/					Clusters the records in the file into k centroids (gpu)
/					Uses squared error check for convergence (gpu)
*/
double* clusterData(Dataset* dataset, int k){

	double* centroids = (double*)calloc(k * (dataset->num_cols+1), sizeof(double));
	double* numeric_records = loadDatasetNumeric(dataset);
	setFirstClusters(centroids, numeric_records, k, dataset->num_rows-1, dataset->num_cols+1);	
	beginGpuClustering(centroids, numeric_records, k, dataset->num_rows-1, dataset->num_cols+1);	
	return centroids;
}

/**
/ loadDatasetNumeric()	Converts the set of strings to double values
/			Loads a flattened double* array with record values
/			Numeric records and cluster records have an additional column (col+1)
*/
double* loadDatasetNumeric(Dataset* dataset){

	int thisrow, thiscol;
	int num_rows = dataset->num_rows;
	int num_cols = dataset->num_cols;
	char** records = dataset->records;
	double *DATASET_NUMERIC;
	
	DATASET_NUMERIC = (double*)calloc(num_rows * (num_cols+1), sizeof(double));

	for(thisrow = 1; thisrow < num_rows; thisrow++){

		for(thiscol = 0; thiscol < num_cols+1; thiscol++){
			int numericIdx = (thisrow-1)*(num_cols+1)+thiscol;
			int recordIdx = thisrow*num_cols+thiscol-1;
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
	printf("\n***\n\tCentroids:\n\n");
	for(i = 0; i < k; i++){
		printf("Centroid [%d]: ", i);
		for(j = 0; j < cols; j++){
			printf("%f ", centroids[i*cols+j]);
		}
		printf("\n");
	}
	free(rands);
}
