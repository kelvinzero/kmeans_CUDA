#include <stdio.h>
#include <stdlib.h>



#ifndef DATASET_H
#define DATASET_H

typedef struct dataset{
	
	char **records;
        int num_rows;
	int num_cols;

}Dataset;


Dataset* initializeDataset(int n_rows, int n_cols);

#endif
