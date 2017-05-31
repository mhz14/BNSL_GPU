#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "config.h"
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

void CheckCudaError(cudaError_t err, char const *errMsg);
#define CUDA_CHECK_RETURN(value1, value2) CheckCudaError(value1, value2)

void calcGPUTimeStart(const char *message);
void calcGPUTimeEnd();

__device__ int findIndex_kernel(int nodesNum, int k, int *combination);

__device__ void findComb_kernel(int nodesNum, int index, int *size,
		int *combination);

__device__ void findCombWithSize_kernel(int nodesNum, int index, int size,
		int* combi);

__device__ void recoverComb_kernel(int vi, int *combi, int size);

__device__ long C_kernel(int n, int m);

__device__ void sortArray_kernel(int *s, int n);

__device__ double calcLocalScore_kernel(int * dev_valuesRange,
		int *dev_samplesValues, int *dev_N, int samplesNum, int* parentSet,
		int size, int curNode, int nodesNum, int valuesMaxNum);

__device__ int binarySearch(double *prob, int ordersNum, double r);

__global__ void calcAllLocalScore_kernel(int *dev_valuesRange,
		int *dev_samplesValues, double *dev_lsTable, int * dev_N,
		int samplesNum, int nodesNum, int allParentSetNumPerNode,
		int valuesMaxNum);

__global__ void generateOrders_kernel(int *dev_newOrders,
		curandState *dev_curandState, int nodesNum, int ordersNum);

__global__ void calcAllOrdersScore_kernel(double *dev_maxLocalScore,
		double *dev_ordersScore, int nodesNum);

__global__ void curandSetup_kernel(curandState *dev_curandState,
		unsigned long long seed);

__global__ void sample_kernel(double *dev_prob, int *dev_samples,
		curandState *dev_curandState, int ordersNum);

__global__ void calcOnePairPerThread_kernel(double * dev_lsTable,
		int * dev_newOrders, double * dev_parentSetScore, int nodesNum,
		int allParentSetNumPerNode, int parentSetNumInOrder);

__global__ void calcMaxParentSetScoreForEachNode_kernel(
		double *dev_parentSetScore, double *dev_maxLocalScore,
		int parentSetNumInOrder, int nodesNum);
