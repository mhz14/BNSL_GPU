#include "config.h"
#include "hp.h"
#include <time.h>
#include <stdio.h>
#include <float.h>
#include <string.h>

void readNodeInfo(int *nodesNum, int **valuesRange);

void readSamples(int **samplesValues, int *samplesNum, int nodesNum);

long C(int n, int m);

void randInitOrder(int *s, int nodesNum);

void selectTwoNodeToSwap(int *n1, int *n2);

void randSwapTwoNode(int *order, int nodesNum);

int compare(const void*a, const void*b);

// calculate max different values number for all pair of child and parent set
int calcValuesMaxNum(int *valuesRange, int nodesNum);

int calcCDF(double *ordersScore, double *prob);

void calcCDFInit(int ordersNum);

void calcCDFFinish();

void startWatch();

int stopWatch();
