#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef FILETOOLS_H
#define FILETOOLS_H



int countFileRows(char *path);
int countFileTokens(char *path);
char* tokenizeLine(char *path);
FILE* loadFile(char *path);


#endif
