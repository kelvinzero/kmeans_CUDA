#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]){

	if(argc != 3)
		exit(-99);

	char *filein = argv[1];
	char *fileout = argv[2];	
	char buffer[255];
	double buffer2[8];

	FILE *fin = fopen(argv[1], "r");
	FILE *fout = fopen(argv[2], "w");

	printf("%s\n%s\n", argv[1], argv[2]);
	if(!fin || !fout){
		printf("Couldnt open file\n");
		exit(-99);
	}
		
	int i;
	fgets(buffer, 255, fin);
	fgets(buffer, 255, fin);
	while(fgets(buffer, 255, fin) != NULL){
		
		if(buffer[strlen(buffer)-1] == '\n')
			buffer[strlen(buffer)-1] = '\0';
			
		sscanf(buffer, "%lf " "%lf " "%lf " "%lf " "%lf " "%lf " "%lf " "%lf", &buffer2[0], &buffer2[1], &buffer2[2], &buffer2[3], &buffer2[4], &buffer2[5], &buffer2[6], &buffer2[7]);
		for(i = 0; i < 8; i++)
			fprintf(fout, "%d %lf\n", i, buffer2[i]);
	} 
	fclose(fin);
	fclose(fout);
	return -1;
}
