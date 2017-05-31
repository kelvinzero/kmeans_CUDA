#include "filetool.h"


int countFileTokens(char *path){

	FILE *readfile;
	char buffer[1024], *strptr = buffer, *test;
	char delims[2] = {',', ' '};
	int columns = 0;
	
	readfile = loadFile(path);
	if(!readfile)
		return 0;
	
	fgets(buffer, 1024, readfile);
	while((test = strtok(strptr, delims)) != NULL){
		strptr = NULL;
		columns++;
	}
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
