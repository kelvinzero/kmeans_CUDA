#include <stdio.h>
#include <string.h>
#include <stdlib.h>


void usage(){

	printf("\n*** Usage:\n\tshrinkdata <in_filename> <out_filename> <lines>\n");
}

int main(int argc, char* argv[]){

	if(argc != 4){
		usage();
		exit(-99);
	}
	int linecount = atoi(argv[3]);
	if(linecount == 0){
		usage();
		exit(-99);
	}
	char *loadfilename = argv[1];
	char *writefilename = argv[2];
	FILE *readfile = fopen(loadfilename, "r");

	if(!readfile){
		printf("\n*** Load file error\n\tCould not load file %s\n", loadfilename);
		exit(-99);
	}
	
	FILE *writefile = fopen(writefilename, "w");
	
	int i;
	char buffer[1024];

	for(i = 0; i <= linecount && fgets(buffer, 1024, readfile); i++){
		fputs(buffer, writefile);		
	}
		
	fclose(writefile);
	fclose(readfile);
}
