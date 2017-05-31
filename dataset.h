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


Dataset* newDataset(int n_rows, int n_cols);
void loadDataset(Dataset* dataset, char* path);

#endif
