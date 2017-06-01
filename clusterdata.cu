#include "clusterdata.h"

double* loadDatasetNumeric(Dataset* dataset);
void setFirstClusters(double* centroids, double* records, int k, int rows, int cols);

/*
/ clusterData()		Converts the dataset to numeric values and clusters
/					Clusters the records in the file into k clusters (gpu)
/					Uses squared error check for convergence (gpu)
*/
double* clusterData(Dataset* dataset, int k){

	double* centroids = (double*)calloc(k * (dataset->num_cols+1), sizeof(double));
	double* numeric_records = loadDatasetNumeric(dataset);
	setFirstClusters(centroids, numeric_records, k, dataset->num_rows-1, dataset->num_cols+1);
	return centroids;
}

/**
/ loadDatasetNumeric()	Converts the set of strings to double values
/						Loads a flattened double* array with record values
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
						Assigned a random record value to each cluster
*/
void setFirstClusters(double* centroids, double* records, int k, int rows, int cols){

	int i,j;
	time_t t;
	srand((unsigned) time(&t));
	int randn;

	// set initial cluster numbers
	for(i = 0; i < k; i++){
		randn = rand() % rows;
		for(j = 0; j < cols; j++){
			if(j==0)
				centroids[i*cols] = i;
			else
				centroids[i*cols+j] = records[randn * cols + j];
		}
	}
}
