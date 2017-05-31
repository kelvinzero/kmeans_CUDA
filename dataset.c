#include "dataset.h"


/**
/ Dataset() 	Createst a dataset
/ 		Creates and allocates memory for a dataset
*/
Dataset* newDataset(int num_cols, int num_rows){

	Dataset* newDataset = (Dataset*)malloc(sizeof(Dataset));
	newDataset->num_cols = num_cols;
	newDataset->num_rows = num_rows;
	newDataset->records = (char**)calloc(num_rows*num_cols, sizeof(char*));
	printf("Dataset initialized\n");
	return newDataset;
}

/**
/ loadDataset()	 Loads data records into dataset
/		 Loads data as a one dimensional array of char pointers
*/
void loadDataset(Dataset* dataset, char* path){


	if(dataset == NULL){
		printf("\n***\n\tNULL dataset error loadDataset()\n\n");
		return;
	}
	
	if(dataset->num_rows == 0 || dataset->num_cols == 0){
		printf("\n***\n\tdataset error loadDataset(): rows or columns = 0\n\n");
		return;
	}	

	FILE *readfile;
	readfile = loadFile(path);

	if(!readfile){
		printf("\n***\tError loading file name: %s\n\n", path);
		return;
	}

	char 	delims[2] = {',', ' '};
	char 	buffer[1024], *bufferptr = buffer;
	int 	thisrow, thiscolumn, num_rows, num_cols;
	char**  records;
	char*   thisVal;

	records  = dataset->records;
	num_rows = dataset->num_rows;
	num_cols = dataset->num_cols;

	for(thisrow = 0; thisrow < num_rows; thisrow++){
			buffer[0] = '\0';
			fgets(buffer, 1024, readfile);	
			bufferptr = buffer;	
		for(thiscolumn = 0; thiscolumn < num_cols; thiscolumn++){
			thisVal = strtok(bufferptr, delims);
			records[thisrow*num_cols+thiscolumn] = (char*)calloc(strlen(thisVal)+1, sizeof(char));
			strcpy(records[thisrow*num_cols+thiscolumn], thisVal);
			bufferptr = NULL;
		}
	} 
}
