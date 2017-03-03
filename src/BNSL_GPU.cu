#include "BNSL_GPU.cuh"

// 节点取值范围
int * valuesRange;

// 结点个数
int nodesNum = 0;

// 样本取值
int * samplesValues;

// 样本数量
int samplesNum;

// 父结点集合的个数
int parentSetNum;

// 局部得分Hash表
double * dev_lsTable;

// 所求结果
int* globalBestGraph;
int* globalBestOrder;
double globalBestScore;

void BNSL_init(){

	// 读取节点信息
	readNodeInfo(&nodesNum, &valuesRange);

	// 读取样本数据
	readSamples(&samplesValues, &samplesNum, nodesNum);

	// 初始化GPU
	CUDA_CHECK_RETURN(cudaDeviceReset(), "cudaDeviceReset failed.");
}

void BNSL_calLocalScore(){

	int i;
	parentSetNum = 0;
	for (i = 0; i <= CONSTRAINTS; i++) {
		parentSetNum = parentSetNum + C(i, nodesNum - 1);
	}

	int * dev_valuesRange;
	int * dev_samplesValues;

	// 在GPU中分配内存空间
	CUDA_CHECK_RETURN(cudaMalloc(&dev_lsTable, nodesNum * parentSetNum * sizeof(double)), "cudaMalloc failed: dev_lsTable.");
	CUDA_CHECK_RETURN(cudaMalloc(&dev_valuesRange, nodesNum * sizeof(int)), "cudaMalloc failed: dev_valuesRange.");
	CUDA_CHECK_RETURN(cudaMalloc(&dev_samplesValues, samplesNum * nodesNum * sizeof(int)), "cudaMalloc failed: dev_samplesValues.");

	// 将数据拷贝到GPU内存中
	CUDA_CHECK_RETURN(cudaMemcpy(dev_valuesRange, valuesRange, nodesNum * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy failed: valuesRange -> dev_valuesRange");
	CUDA_CHECK_RETURN(cudaMemcpy(dev_samplesValues, samplesValues, samplesNum * nodesNum * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy failed: samplesValues -> dev_samplesValues");

	// 启动GPU计算
	int threadNum = 64;
	int total = parentSetNum * nodesNum;
	int blockNum = (total - 1) / threadNum + 1;
	calcAllPossibleLocalScore_kernel << <blockNum, threadNum >> >(dev_valuesRange, dev_samplesValues, dev_lsTable, samplesNum, nodesNum, parentSetNum);
	CUDA_CHECK_RETURN(cudaGetLastError(), "calcAllPossibleLocalScore_kernel launch failed.");
	CUDA_CHECK_RETURN(cudaDeviceSynchronize(), "calcAllPossibleLocalScore_kernel failed on running.");

	// 释放在GPU中分配的内存空间
	CUDA_CHECK_RETURN(cudaFree(dev_valuesRange), "cudaFree failed: dev_valuesRange.");
	CUDA_CHECK_RETURN(cudaFree(dev_samplesValues), "cudaFree failed: dev_samplesValues.");

	// 回收存放样本数据的内存
	free(valuesRange);
	free(samplesValues);
}

void BNSL_start(){

	int i, j, iter;
	int parentSetNumInOrder = 0;
	for (i = 0; i < nodesNum; i++){
		for (j = 0; j <= CONSTRAINTS&&j < i + 1; j++){
			parentSetNumInOrder += C(j, i);
		}
	}

	// 每次新产生63个order，加上1个初试order
	int ordersNum = 128;

	// 迭代次数
	int iterNum = 0;

	// 初始化随机函数的随机种子
	srand((unsigned int)time(NULL));

	// 随机种子
	int seed = 1234;

	// GPU中存储新产生的拓扑排序
	int * dev_newOrders;
	CUDA_CHECK_RETURN(cudaMalloc(&dev_newOrders, ordersNum * nodesNum * sizeof(int)), "cudaMalloc failed: dev_newOrders.");
	// CPU中存储拓扑排序
	int * newOrder = (int *)malloc(nodesNum * sizeof(int));
	// 初始化拓扑排序
	randInitOrder(newOrder, nodesNum);

	// GPU中存储符合拓扑排序的父节点集合的得分
	double * dev_parentSetScore;
	CUDA_CHECK_RETURN(cudaMalloc(&dev_parentSetScore, ordersNum * parentSetNumInOrder * sizeof(double)), "cudaMalloc failed: dev_result.");

	// GPU中存储每个结点父结点集合的最高得分
	double * dev_maxLocalScore;
	CUDA_CHECK_RETURN(cudaMalloc(&dev_maxLocalScore, ordersNum * nodesNum * sizeof(double)), "cudaMalloc failed: dev_maxLocalScore.");

	// GPU中存储每个拓扑排序的得分
	double * dev_ordersScore;
	CUDA_CHECK_RETURN(cudaMalloc(&dev_ordersScore, ordersNum * sizeof(double)), "cudaMalloc failed: dev_ordersScore.");
	// CPU中存储每个拓扑排序的得分
	double * ordersScore = (double *)malloc(ordersNum * sizeof(double));

	// GPU中存储I的概率分布
	double *dev_prob;
	CUDA_CHECK_RETURN(cudaMalloc(&dev_prob, ordersNum * sizeof(double)), "cudaMalloc failed: dev_prob.");
	// CPU中I的概率分布
	double *prob = (double *)malloc(ordersNum * sizeof(double));

	// GPU中存储I的样本
	int *dev_samples;
	CUDA_CHECK_RETURN(cudaMalloc(&dev_samples, ordersNum * sizeof(int)), "cudaMalloc failed: dev_samples.");
	// CPU中I的样本
	int *samples = (int *)malloc(ordersNum * sizeof(int));

	// CPU中存储全局最优的拓扑排序
	globalBestOrder = (int *)malloc(nodesNum * sizeof(int));
	globalBestScore = -FLT_MAX;

	// GPU中存储curand的随机状态
	curandState *dev_curandState;
	CUDA_CHECK_RETURN(cudaMalloc(&dev_curandState, ordersNum * sizeof(curandState)), "cudaMalloc failed: dev_curandState.");
	// 初始化curand的随机状态
	curandSetup_kernel << < 1, ordersNum >> >(dev_curandState, seed);
	CUDA_CHECK_RETURN(cudaGetLastError(), "curandSetup_kernel launch failed.");

	calcCDFInit(ordersNum);

	for (iter = 0; iter < iterNum; iter++){
		// 随机产生新的拓扑排序
		CUDA_CHECK_RETURN(cudaMemcpy(dev_newOrders, newOrder, nodesNum * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy failed: newOrder -> dev_newOrders.");
		generateOrders_kernel << <1, ordersNum, nodesNum * 4 >> >(dev_newOrders, dev_curandState, nodesNum);
		CUDA_CHECK_RETURN(cudaGetLastError(), "generateOrders_kernel launch failed.");

		//calcGPUTimeStart("calcOnePairPerThread_kernel: ");
		int totalPairNum = ordersNum * parentSetNumInOrder;
		int threadDimX = 128;
		int blockDim = (totalPairNum - 1) / threadDimX + 1;
		int blockDimX = 1;
		int blockDimY = 1;
		if (blockDim < 65535){
			blockDimX = 1;
			blockDimY = blockDim;
		}
		else{
			blockDimX = (blockDim - 1) / 65535 + 1;
			blockDimY = 65535;
		}
		dim3 gridDim(blockDimX, blockDimY);
		calcOnePairPerThread_kernel << <gridDim, threadDimX >> >(dev_lsTable, dev_newOrders, dev_parentSetScore, nodesNum, parentSetNum, parentSetNumInOrder);
		CUDA_CHECK_RETURN(cudaGetLastError(), "calcOnePairPerThread_kernel launch failed.");
		//calcGPUTimeEnd();

		// 查找每个结点得分最高的父结点集合
		calcMaxParentSetScoreForEachNode_kernel << <nodesNum, ordersNum >> >(dev_parentSetScore, dev_maxLocalScore, parentSetNumInOrder, nodesNum);
		CUDA_CHECK_RETURN(cudaGetLastError(), "calcMaxLocalScoreForEachNode_kernel launch failed.");

		// 计算所有拓扑排序的得分
		calcAllOrdersScore_kernel << <1, ordersNum >> >(dev_maxLocalScore, dev_ordersScore, nodesNum);
		CUDA_CHECK_RETURN(cudaGetLastError(), "calcAllOrdersScore_kernel launch failed.");
		CUDA_CHECK_RETURN(cudaMemcpy(ordersScore, dev_ordersScore, ordersNum * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy failed: dev_ordersScore -> ordersScore.");

		int *newOrders = (int *)malloc(ordersNum * nodesNum * sizeof(int));
		CUDA_CHECK_RETURN(cudaMemcpy(newOrders, dev_newOrders, ordersNum * nodesNum * sizeof(int), cudaMemcpyDeviceToHost), "test");

		// 将拓扑排序的得分转化为I的累积概率分布
		int maxId = calcCDF(ordersScore, prob);

		// 与最优解比较
		if (ordersScore[maxId] > globalBestScore){
			CUDA_CHECK_RETURN(cudaMemcpy(globalBestOrder, dev_newOrders + maxId * nodesNum, nodesNum * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy failed: dev_newOrders -> globalBestOrder");
			globalBestScore = ordersScore[maxId];
		}

		// 对辅助变量I取样
		CUDA_CHECK_RETURN(cudaMemcpy(dev_prob, prob, ordersNum * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy failed: prob -> dev_prob.");
		sample_kernel << <1, ordersNum, ordersNum * 8 >> >(dev_prob, dev_samples, dev_curandState, ordersNum);
		CUDA_CHECK_RETURN(cudaGetLastError(), "sample_kernel launch failed.");
		CUDA_CHECK_RETURN(cudaMemcpy(samples, dev_samples, ordersNum * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy failed: dev_samples -> samples.");

		int r = rand() % ordersNum;
		CUDA_CHECK_RETURN(cudaMemcpy(newOrder, dev_newOrders + samples[r] * nodesNum, nodesNum * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy failed: dev_newOrders -> newOrder");
	}

	CUDA_CHECK_RETURN(cudaFree(dev_newOrders), "cudaFree failed: dev_newOrders.");
	CUDA_CHECK_RETURN(cudaFree(dev_parentSetScore), "cudaFree failed: dev_parentSetScore.");
	CUDA_CHECK_RETURN(cudaFree(dev_maxLocalScore), "cudaFree failed: dev_maxLocalScore.");
	CUDA_CHECK_RETURN(cudaFree(dev_ordersScore), "cudaFree failed: dev_ordersScore.");
	CUDA_CHECK_RETURN(cudaFree(dev_prob), "cudaFree failed: dev_prob.");
	CUDA_CHECK_RETURN(cudaFree(dev_samples), "cudaFree failed: dev_samples.");
	CUDA_CHECK_RETURN(cudaFree(dev_curandState), "cudaFree failed: dev_curandState.");
	free(newOrder);
	free(ordersScore);
	free(prob);
	free(samples);
	calcCDFFinish();
}

void BNSL_printResult() {
	/*
	printf("Bayesian Network learned:\n");
	for (int i = 0; i < nodesNum; i++){
		for (int j = 0; j < nodesNum; j++){
			printf("%d ", globalBestGraph[i*nodesNum + j]);
		}
		printf("\n");
	}
	*/

	printf("Best Score: %f \n", globalBestScore);
	printf("Best Topology: ");
	for (int i = 0; i < nodesNum; i++){
		printf("%d ", globalBestOrder[i]);
	}
	printf("\n");
}

void BNSL_finish(){
	CUDA_CHECK_RETURN(cudaFree(dev_lsTable), "cudaFree failed: dev_lsTable.");
	free(globalBestOrder);
	free(globalBestGraph);
}