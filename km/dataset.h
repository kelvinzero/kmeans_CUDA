#include <stdio.h>
#include <stdlib.h>
#include "filetool.h"


#ifndef DATASET_H
#define DATASET_H

typedef struct dataset{
	
	char **records;
        int num_rows;
	int num_cols;

}Dataset;

void freeDataset(Dataset* dataset);
Dataset* newDataset(int n_rows, int n_cols);
void loadDataset(Dataset* dataset, char* path);
int  testDataset(Dataset* dataset, int num_rows, int num_cols);
void writeData(Dataset* dataset, char* path);
#endif
