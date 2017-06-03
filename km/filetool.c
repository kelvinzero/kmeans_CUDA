#include "filetool.h"


int countFileTokens(char *path){

	FILE *readfile;
	char buffer[1024], *strptr = buffer, *test;
	const char *delims = " ";
	int columns = 0;
	
	readfile = loadFile(path, "r");
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

	readfile = loadFile(path, "r");
	if(!readfile)
		return 0;

	for(rowcount = 0; fgets(buffer, 1024, readfile); rowcount++);
	
	fclose(readfile);
	return rowcount;
}

FILE* loadFile(char *path, const char * mode){
	
	FILE *read_file;
	if(!(read_file = fopen(path, mode))){
		return NULL;
	}
	return read_file;
}
