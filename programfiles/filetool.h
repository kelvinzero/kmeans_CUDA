#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef FILETOOLS_H
#define FILETOOLS_H



int countFileRows(char *path);
int countFileTokens(char *path);
char* tokenizeLine(char *path);
FILE* loadFile(char *path);
void writeClusterFiles(char *OUT_PREFIX, double *NUMERIC_RECORDS, double *CLUSTERS, int K, int ROWS, int COLUMNS);

#endif
