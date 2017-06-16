#include "filetool.h"


int countFileTokens(char *path){

	FILE *readfile;
	char buffer[1024], *strptr = buffer, *test;
	const char *delims = " ";
	int columns = 0;
	
	readfile = loadFile(path);
	if(!readfile)
		return 0;
	
	fgets(buffer, 1024, readfile);
	while((test = strtok(strptr, delims)) != NULL){
		strptr = NULL;
		columns++;
	}
	fclose(readfile);
	return columns;
}

int countFileRows(char *path){

	FILE* readfile = NULL;
	char buffer[1024];
	int rowcount;

	readfile = loadFile(path);
	if(!readfile)
		return 0;

	for(rowcount = 0; fgets(buffer, 1024, readfile); rowcount++);
	
	fclose(readfile);
	return rowcount;
}

FILE* loadFile(char *path){
	
	FILE *read_file;
	if(!(read_file = fopen(path, "r"))){
		return NULL;
	}
	return read_file;
}


void writeClusterFiles(char *OUT_PREFIX, double *NUMERIC_RECORDS, double *CLUSTERS, int K, int ROWS, int COLUMNS){

	int i,j,processed;
	char fnameBuffer[255];
	char numHolder[10];

	printf("\nWriting cluster files...\n\n");
	
	FILE ** files = (FILE**)malloc(K * sizeof(FILE*));
	for(i = 0; i < K; i++){
		fnameBuffer[0] = '\0';
		sprintf(numHolder, "%d",  i+1);
		strcpy(fnameBuffer, OUT_PREFIX);
		strcat(fnameBuffer, ".CL");
		strcat(fnameBuffer, numHolder);
		files[i] = fopen(fnameBuffer, "w");
	}
	
	for(i = 0; i < K; i++){
		sprintf(numHolder, "%d", i+1);
		fputs("## Cluster:  ", files[i]);
		fputs(numHolder, files[i]);
		sprintf(numHolder, "%d", (int)CLUSTERS[i*COLUMNS]);
		fputs("  Records: ", files[i]);
		fputs(numHolder, files[i]);
		fputs("\n## Centroid means: ", files[i]);
		for(j = 1; j < COLUMNS; j++){
			sprintf(numHolder, "%lf", CLUSTERS[i*COLUMNS+j]);
			fputs(numHolder, files[i]);
			fputs(" ", files[i]);
		}	
		fputs("\n\n", files[i]);
		
	}

	int recordCentroid;
	for(i = 0; i < ROWS; i++){
		recordCentroid = (int)NUMERIC_RECORDS[i*COLUMNS]; 
		for(j = 1; j < COLUMNS; j++){
			sprintf(numHolder, "%lf", NUMERIC_RECORDS[i*COLUMNS+j]);
			fputs(numHolder, files[recordCentroid]);
			fputs(" ", files[recordCentroid]);
		}
		fputs("\n", files[recordCentroid]);
	}	
	for(i = 0; i < K; i++)
		fclose(files[i]);
	free(files);
}
