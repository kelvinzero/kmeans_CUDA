#include "filetool.h"


int countFileTokens(char *path){

	FILE *readfile;
	char buffer[1024];
	int columns;

	return 1;
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
