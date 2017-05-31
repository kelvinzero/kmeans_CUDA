#include<stdio.h>
#include<stdlib.h>
#include "dataset.h"
#include "filetool.h"

#ifndef K_MEANSCLUSTERING_H
#define K_MEANSCLUSTERING_H

Dataset* DATASET; 

int ROWS;
int COLUMNS;
int K;

char *INFILE_NAME;
char *OUTFILE_PREFIX;

void usage();

#endif
