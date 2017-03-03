#include "config.h"
#include "hp.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

void readNodeInfo(int *nodesNum, int **valuesRange);

void readSamples(int **samplesValues, int *samplesNum, int nodesNum);

long C(int n, int m);

void randInitOrder(int *s, int nodesNum);

int calcCDF(double *ordersScore, double *prob);

void calcCDFInit(int ordersNum);

void calcCDFFinish();

void calcCPUTimeStart(char const *message);
void calcCPUTimeEnd();
