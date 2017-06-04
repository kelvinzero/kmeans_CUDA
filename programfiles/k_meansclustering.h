#include<stdio.h>
#include<stdlib.h>
#include "dataset.h"
#include "filetool.h"
#include "clusterdata.h"

#ifndef K_MEANSCLUSTERING_H
#define K_MEANSCLUSTERING_H

Dataset* DATASET; 
double **CLUSTERS; 
double *NUMERIC_RECORDS;

int ROWS;
int COLUMNS;
int K;

char *INFILE_NAME;
char *OUTFILE_PREFIX;

void usage();

#endif
