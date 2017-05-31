#include "BNSL_GPU.cuh"

int *valuesRange, *samplesValues;

int nodesNum, samplesNum;

int allParentSetNumPerNode;

double * dev_lsTable;

int* globalBestGraph, *globalBestOrder;
double globalBestScore;

int initTime, calcLocalScoreTime, searchTime;

void BNSL_init() {
	startWatch();
	readNodeInfo(&nodesNum, &valuesRange);
	readSamples(&samplesValues, &samplesNum, nodesNum);
	initTime = stopWatch();
}

void BNSL_calcLocalScore() {
	startWatch();

	allParentSetNumPerNode = 0;
	for (int i = 0; i <= CONSTRAINTS; i++) {
		allParentSetNumPerNode = allParentSetNumPerNode + C(i, nodesNum - 1);
	}

	int * dev_valuesRange;
	int * dev_samplesValues;
	int * dev_N;

	// calculate max different values number for all pair of child and parent set
	int valuesMaxNum = calcValuesMaxNum(valuesRange, nodesNum);

	CUDA_CHECK_RETURN(
			cudaMalloc(&dev_lsTable,
					nodesNum * allParentSetNumPerNode * sizeof(double)),
			"cudaMalloc failed: dev_lsTable.");
	CUDA_CHECK_RETURN(cudaMalloc(&dev_valuesRange, nodesNum * sizeof(int)),
			"cudaMalloc failed: dev_valuesRange.");
	CUDA_CHECK_RETURN(
			cudaMalloc(&dev_samplesValues, samplesNum * nodesNum * sizeof(int)),
			"cudaMalloc failed: dev_samplesValues.");
	CUDA_CHECK_RETURN(
			cudaMalloc(&dev_N,
					valuesMaxNum * allParentSetNumPerNode * sizeof(int)),
			"dev_N cudaMalloc failed.");

	CUDA_CHECK_RETURN(
			cudaMemcpy(dev_valuesRange, valuesRange, nodesNum * sizeof(int),
					cudaMemcpyHostToDevice),
			"cudaMemcpy failed: valuesRange -> dev_valuesRange");
	CUDA_CHECK_RETURN(
			cudaMemcpy(dev_samplesValues, samplesValues,
					samplesNum * nodesNum * sizeof(int),
					cudaMemcpyHostToDevice),
			"cudaMemcpy failed: samplesValues -> dev_samplesValues");

	int threadNum = 256;
	int blockNum = (allParentSetNumPerNode - 1) / threadNum + 1;
//	calcAllLocalScore_kernel<<<blockNum, threadNum, nodesNum * sizeof(int)>>>(
//			dev_valuesRange, dev_samplesValues, dev_lsTable, dev_N, samplesNum,
//			nodesNum, allParentSetNumPerNode, valuesMaxNum);

	CUDA_CHECK_RETURN(cudaFree(dev_valuesRange),
			"cudaFree failed: dev_valuesRange.");
	CUDA_CHECK_RETURN(cudaFree(dev_samplesValues),
			"cudaFree failed: dev_samplesValues.");
	CUDA_CHECK_RETURN(cudaFree(dev_N), "cudaFree failed: dev_N.");

	free(valuesRange);
	free(samplesValues);
	calcLocalScoreTime = stopWatch();
}

void BNSL_search() {
	startWatch();

	int i, j, iter;
	int parentSetNumInOrder = 0;
	for (i = 0; i < nodesNum; i++) {
		for (j = 0; j <= CONSTRAINTS && j < i + 1; j++) {
			parentSetNumInOrder += C(j, i);
		}
	}

	int ordersNum = 128;

	int iterNum = ITER;

	srand((unsigned int) time(NULL));

	int seed = 1234;

	int * dev_newOrders;
	CUDA_CHECK_RETURN(
			cudaMalloc(&dev_newOrders, ordersNum * nodesNum * sizeof(int)),
			"cudaMalloc failed: dev_newOrders.");

	int * newOrder = (int *) malloc(nodesNum * sizeof(int));
	CUDA_CHECK_RETURN(cudaMallocHost(&newOrder, nodesNum * sizeof(int)),
			"cudaMallocHost failed: newOrder.");

	randInitOrder(newOrder, nodesNum);

	double * dev_parentSetScore;
	CUDA_CHECK_RETURN(
			cudaMalloc(&dev_parentSetScore,
					ordersNum * parentSetNumInOrder * sizeof(double)),
			"cudaMalloc failed: dev_parentSetScore.");

	double * dev_maxLocalScore;
	CUDA_CHECK_RETURN(
			cudaMalloc(&dev_maxLocalScore,
					ordersNum * nodesNum * sizeof(double)),
			"cudaMalloc failed: dev_maxLocalScore.");

	double * dev_ordersScore, *ordersScore;
	CUDA_CHECK_RETURN(cudaMalloc(&dev_ordersScore, ordersNum * sizeof(double)),
			"cudaMalloc failed: dev_ordersScore.");
	CUDA_CHECK_RETURN(cudaMallocHost(&ordersScore, ordersNum * sizeof(double)),
			"cudaMallocHost failed: ordersScore.");

	double *dev_prob, *prob;
	CUDA_CHECK_RETURN(cudaMalloc(&dev_prob, ordersNum * sizeof(double)),
			"cudaMalloc failed: dev_prob.");
	CUDA_CHECK_RETURN(cudaMallocHost(&prob, ordersNum * sizeof(double)),
			"cudaMallocHost failed: prob.");

	int *dev_samples, *samples;
	CUDA_CHECK_RETURN(cudaMalloc(&dev_samples, ordersNum * sizeof(int)),
			"cudaMalloc failed: dev_samples.");
	CUDA_CHECK_RETURN(cudaMallocHost(&samples, ordersNum * sizeof(int)),
			"cudaMallocHost failed: samples.");

	globalBestOrder = (int *) malloc(nodesNum * sizeof(int));
	globalBestScore = -FLT_MAX;

	curandState *dev_curandState;
	CUDA_CHECK_RETURN(
			cudaMalloc(&dev_curandState, ordersNum * sizeof(curandState)),
			"cudaMalloc failed: dev_curandState.");

	curandSetup_kernel<<<1, ordersNum>>>(dev_curandState, seed);
	CUDA_CHECK_RETURN(cudaGetLastError(), "curandSetup_kernel launch failed.");

	calcCDFInit(ordersNum);

	for (iter = 1; iter <= iterNum; iter++) {
		printf("iter = %d:\n", iter);

		//calcGPUTimeStart("generateOrders_kernel: ");
		CUDA_CHECK_RETURN(
				cudaMemcpy(dev_newOrders, newOrder, nodesNum * sizeof(int),
						cudaMemcpyHostToDevice),
				"cudaMemcpy failed: newOrder -> dev_newOrders.");
		generateOrders_kernel<<<1, 128, nodesNum * sizeof(int)>>>(dev_newOrders,
				dev_curandState, nodesNum, ordersNum);
		CUDA_CHECK_RETURN(cudaGetLastError(),
				"generateOrders_kernel launch failed.");
		//calcGPUTimeEnd();

		//calcGPUTimeStart("calcOnePairPerThread_kernel: ");
		int threadNum = 128;
		int blockNum = (parentSetNumInOrder - 1) / threadNum + 1;
		dim3 gridDim(blockNum, ordersNum);
		calcOnePairPerThread_kernel<<<gridDim, threadNum>>>(dev_lsTable,
				dev_newOrders, dev_parentSetScore, nodesNum,
				allParentSetNumPerNode, parentSetNumInOrder);
		CUDA_CHECK_RETURN(cudaGetLastError(),
				"calcOnePairPerThread_kernel launch failed.");
		//calcGPUTimeEnd();

		//calcGPUTimeStart("calcMaxParentSetScoreForEachNode_kernel: ");
		calcMaxParentSetScoreForEachNode_kernel<<<nodesNum, ordersNum>>>(
				dev_parentSetScore, dev_maxLocalScore, parentSetNumInOrder,
				nodesNum);
		CUDA_CHECK_RETURN(cudaGetLastError(),
				"calcMaxLocalScoreForEachNode_kernel launch failed.");
		//calcGPUTimeEnd();

		calcAllOrdersScore_kernel<<<1, ordersNum>>>(dev_maxLocalScore,
				dev_ordersScore, nodesNum);
		CUDA_CHECK_RETURN(cudaGetLastError(),
				"calcAllOrdersScore_kernel launch failed.");
		CUDA_CHECK_RETURN(
				cudaMemcpy(ordersScore, dev_ordersScore,
						ordersNum * sizeof(double), cudaMemcpyDeviceToHost),
				"cudaMemcpy failed: dev_ordersScore -> ordersScore.");

		int maxId = calcCDF(ordersScore, prob);

		if (ordersScore[maxId] > globalBestScore) {
			CUDA_CHECK_RETURN(
					cudaMemcpy(globalBestOrder,
							dev_newOrders + maxId * nodesNum,
							nodesNum * sizeof(int), cudaMemcpyDeviceToHost),
					"cudaMemcpy failed: dev_newOrders -> globalBestOrder");
			globalBestScore = ordersScore[maxId];
		}

		CUDA_CHECK_RETURN(
				cudaMemcpy(dev_prob, prob, ordersNum * sizeof(double),
						cudaMemcpyHostToDevice),
				"cudaMemcpy failed: prob -> dev_prob.");
		sample_kernel<<<1, ordersNum, ordersNum * 8>>>(dev_prob, dev_samples,
				dev_curandState, ordersNum);
		CUDA_CHECK_RETURN(cudaGetLastError(), "sample_kernel launch failed.");
		CUDA_CHECK_RETURN(
				cudaMemcpy(samples, dev_samples, ordersNum * sizeof(int),
						cudaMemcpyDeviceToHost),
				"cudaMemcpy failed: dev_samples -> samples.");

		int r = rand() % ordersNum;
		CUDA_CHECK_RETURN(
				cudaMemcpy(newOrder, dev_newOrders + samples[r] * nodesNum,
						nodesNum * sizeof(int), cudaMemcpyDeviceToHost),
				"cudaMemcpy failed: dev_newOrders -> newOrder");
	}

	CUDA_CHECK_RETURN(cudaFree(dev_newOrders),
			"cudaFree failed: dev_newOrders.");
	CUDA_CHECK_RETURN(cudaFree(dev_parentSetScore),
			"cudaFree failed: dev_parentSetScore.");
	CUDA_CHECK_RETURN(cudaFree(dev_maxLocalScore),
			"cudaFree failed: dev_maxLocalScore.");
	CUDA_CHECK_RETURN(cudaFree(dev_ordersScore),
			"cudaFree failed: dev_ordersScore.");
	CUDA_CHECK_RETURN(cudaFree(dev_prob), "cudaFree failed: dev_prob.");
	CUDA_CHECK_RETURN(cudaFree(dev_samples), "cudaFree failed: dev_samples.");
	CUDA_CHECK_RETURN(cudaFree(dev_curandState),
			"cudaFree failed: dev_curandState.");
	CUDA_CHECK_RETURN(cudaFreeHost(newOrder), "cudaFreeHost failed: newOrder.");
	CUDA_CHECK_RETURN(cudaFreeHost(ordersScore),
			"cudaFreeHost failed: ordersScore.");
	CUDA_CHECK_RETURN(cudaFreeHost(prob), "cudaFreeHost failed: prob.");
	CUDA_CHECK_RETURN(cudaFreeHost(samples), "cudaFreeHost failed: samples.");

	calcCDFFinish();

	searchTime = stopWatch();
}

void BNSL_printResult() {

	printf("Best Score: %f \n", globalBestScore);
	printf("Best Topology: ");
	for (int i = 0; i < nodesNum; i++) {
		printf("%d ", globalBestOrder[i]);
	}
	printf("\n");
	printf("BNSL_init elapsed time is %dms.\n", initTime);
	printf("BNSL_calcLocalScore time is %dms. \n", calcLocalScoreTime);
	printf("BNSL_search time is %dms. \n", searchTime);
}

void BNSL_finish() {
	CUDA_CHECK_RETURN(cudaFree(dev_lsTable), "cudaFree failed: dev_lsTable.");
	free(globalBestOrder);
	free(globalBestGraph);
}
