#include "dataset.h"


/**
/ Dataset() 	- Createst a dataset
/ 		Creates and allocates memory for a dataset
*/
Dataset* initializeDataset(int num_cols, int num_rows){

	Dataset* newDataset = (Dataset*)malloc(sizeof(Dataset));
	newDataset->num_cols = num_cols;
	newDataset->num_rows = num_rows;
	newDataset->records = (char**)calloc(num_rows, sizeof(char*));
	return newDataset;
}
