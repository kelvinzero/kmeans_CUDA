#include "k_meansclustering.h"

#define bool int
#define TRUE 0
#define FALSE 1

int main(int argc, char *argv[]){
	
	char* infile_name;

	if(argc != 4){
		usage();
		exit(-1);
	}
	
	INFILE_NAME 	= argv[1];	
	K 		= atoi(argv[3]);	
	
	INFILE_NAME 	= argv[1];
	OUTFILE_PREFIX 	= argv[2];

	ROWS = countFileRows(INFILE_NAME);
	COLUMNS = countFileTokens(INFILE_NAME);

	if(K == 0){
		usage();
		exit(-1);	
	}
	if(ROWS == 0 || COLUMNS == 0){
		printf("\n***\n\tError loading file - incorrect format - File: %s\n\n", INFILE_NAME);
		exit(-1);
	}
		
	printf("\n***\n\tLoaded Filename: %s\n\tRows: %d\n\n", INFILE_NAME, ROWS); 

	return 1;
}


FILE* loadDataset(char *path){
	return NULL;

}

void usage(){
	printf("\n**Usage:*\n\tkmeans [in_filename] [out_filenameprefix] [k clusters > 0]\n\n" );
}
