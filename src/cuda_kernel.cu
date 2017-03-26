#include "cuda_kernel.cuh"

cudaEvent_t start, stop;

void CheckCudaError(cudaError_t err, char const* errMsg) {
	if (err == cudaSuccess)
		return;
	printf("%s\nError Message: %s.\n", errMsg, cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}

void calcGPUTimeStart(const char *message) {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, NULL);
	printf("%s", message);
}

void calcGPUTimeEnd() {
	float time = 0;
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("Elapsed GPU time is %fms\n", time);
}

// ----device kernel----
__device__ int findIndex_kernel(int nodesNum, int k, int* combi) {
	int index = 1;
	int i, j;

	for (j = 1; j < combi[0]; j++) {
		index += C_kernel(k - 1, nodesNum - 1 - j);
	}
	for (i = 2; i <= k; i++) {
		for (j = combi[i - 2] + 1; j < combi[i - 1]; j++) {
			index = index + C_kernel(k - i, nodesNum - 1 - j);
		}
	}

	for (i = 1; i < k; i++) {
		index = index + C_kernel(i, nodesNum - 1);
	}

	return index;
}

__device__ void findComb_kernel(int nodesNum, int index, int* size,
		int* combi) {

	if (index == 0) {
		*size = 0;
	} else {
		int k = 1;
		int limit = C_kernel(k, nodesNum - 1);
		while (index > limit) {
			k++;
			limit = limit + C_kernel(k, nodesNum - 1);
		}
		index = index - limit + C_kernel(k, nodesNum - 1);
		*size = k;

		int base = 0;
		int n = nodesNum - 1;
		int i, sum, shift;
		int sum_new = 0;

		for (i = 1; i < k; i++) {
			sum = 0;
			for (shift = 1; shift <= n; shift++) {
				sum_new = sum + C_kernel(k - i, n - shift);
				if (sum_new < index) {
					sum = sum_new;
				} else {
					break;
				}
			}
			combi[i - 1] = base + shift;
			n = n - shift;
			index = index - sum;
			base = combi[i - 1];
		}
		combi[k - 1] = base + index;
	}
}

__device__ void findCombWithSize_kernel(int nodesNum, int index, int size,
		int* combi) {

	if (index != 0 && size != 0) {
		int base = 0;
		int n = nodesNum - 1;
		int i, sum, shift;
		int sum_new = 0;

		for (i = 1; i < size; i++) {
			sum = 0;
			for (shift = 1; shift <= n; shift++) {
				sum_new = sum + C_kernel(size - i, n - shift);
				if (sum_new < index) {
					sum = sum_new;
				} else {
					break;
				}
			}
			combi[i - 1] = base + shift;
			n = n - shift;
			index = index - sum;
			base = combi[i - 1];
		}
		combi[size - 1] = base + index;
	}
}

__device__ void recoverComb_kernel(int curNode, int* combi, int size) {

	for (int i = 0; i < size; i++) {
		if (combi[i] >= curNode + 1) {
			combi[i] = combi[i] + 1;
		}
	}
}

__device__ long C_kernel(int n, int m) {

	if (n > m || n < 0 || m < 0)
		return -1;

	int k, res = 1;
	for (k = 1; k <= n; k++) {
		res = (res * (m - n + k)) / k;
	}
	return res;
}

__device__ void sortArray_kernel(int * s, int n) {
	int min;
	int id = -1;
	for (int i = 0; i < n - 1; i++) {
		min = INT_MAX;
		id = -1;
		for (int j = i; j < n; j++) {
			if (s[j] < min) {
				min = s[j];
				id = j;
			}
		}
		int swap = s[i];
		s[i] = s[id];
		s[id] = swap;
	}
}

__device__ double calLocalScore_kernel(int * dev_valuesRange,
		int *dev_samplesValues, int *dev_N, int samplesNum, int* parentSet, int size,
		int curNode, int nodesNum, int valuesMaxNum) {

	int curNodeValuesNum = dev_valuesRange[curNode];
	int valuesNum = 1;
	int i, j;
	for (i = 0; i < size; i++) {
		valuesNum = valuesNum * dev_valuesRange[parentSet[i] - 1];
	}

	int *N = dev_N + (blockIdx.x * blockDim.x + threadIdx.x) * valuesMaxNum;
	int pvalueIndex = 0;
	for (i = 0; i < samplesNum; i++) {
		pvalueIndex = 0;
		for (j = 0; j < size; j++) {
			pvalueIndex = pvalueIndex * dev_valuesRange[parentSet[j] - 1]
					+ dev_samplesValues[i * nodesNum + parentSet[j] - 1];
		}
		N[pvalueIndex * curNodeValuesNum
				+ dev_samplesValues[i * nodesNum + curNode]]++;

	}

	double alpha = ALPHA / (dev_valuesRange[curNode] * valuesNum);
	double localScore = size * log(GAMMA);
	for (i = 0; i < valuesNum; i++) {
		int sum = 0;
		for (j = 0; j < curNodeValuesNum; j++) {
			int cur = i * curNodeValuesNum + j;
			if (N[cur] != 0) {
				localScore = localScore + lgamma(N[cur] + alpha)
						- lgamma(alpha);
				sum = sum + N[cur];
			}
		}
		localScore = localScore + lgamma(alpha * curNodeValuesNum)
				- lgamma(alpha * curNodeValuesNum + sum);
	}

	return localScore;
}

__device__ int binarySearch(double *prob, int ordersNum, double r) {
	int start = 0, end = ordersNum - 1;
	int mid;
	while (start <= end) {
		mid = (start + end) / 2;
		if (abs(r - prob[mid]) < 1e-300) {
			return mid;
		} else if (r > prob[mid]) {
			start = mid + 1;
		} else {
			end = mid - 1;
		}
	}
	return start;
}

// ----global kernel----

__global__ void calcAllLocalScore_kernel(int *dev_valuesRange,
		int *dev_samplesValues, int *dev_N, double *dev_lsTable, int samplesNum,
		int nodesNum, int allParentSetNumPerNode, int valuesMaxNum) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int limit = nodesNum * allParentSetNumPerNode;
	if (id < limit) {
		int nodeId = id / allParentSetNumPerNode;
		int parentSetId = id % allParentSetNumPerNode;

		int parentSet[CONSTRAINTS];
		int size = 0;

		findComb_kernel(nodesNum, parentSetId, &size, parentSet);

		recoverComb_kernel(nodeId, parentSet, size);

		dev_lsTable[id] = calLocalScore_kernel(dev_valuesRange,
				dev_samplesValues, dev_N, samplesNum, parentSet, size, nodeId,
				nodesNum, valuesMaxNum);
	}
}

__global__ void generateOrders_kernel(int *dev_newOrders,
		curandState *dev_curandState, int nodesNum) {
	extern __shared__ int initialOrder[];
	if (threadIdx.x < nodesNum) {
		initialOrder[threadIdx.x] = dev_newOrders[threadIdx.x];
	}
	__syncthreads();

	int i, j, k;
	for (i = 0; i < nodesNum; i++) {
		dev_newOrders[threadIdx.x * nodesNum + i] = initialOrder[i];
	}

	if (threadIdx.x != 0) {
		curandState localState = dev_curandState[threadIdx.x];
		i = curand(&localState) % nodesNum;
		j = curand(&localState) % nodesNum;
		while (i == j) {
			j = curand(&localState) % nodesNum;
		}
		dev_curandState[threadIdx.x] = localState;

		k = dev_newOrders[threadIdx.x * nodesNum + i];
		dev_newOrders[threadIdx.x * nodesNum + i] = dev_newOrders[threadIdx.x
				* nodesNum + j];
		dev_newOrders[threadIdx.x * nodesNum + j] = k;
	}
}

__global__ void calcAllOrdersScore_kernel(double *dev_maxLocalScore,
		double *dev_ordersScore, int nodesNum) {
	double sum = 0.0;
	int i;
	for (i = 0; i < nodesNum; i++) {
		sum += dev_maxLocalScore[threadIdx.x * nodesNum + i];
	}
	dev_ordersScore[threadIdx.x] = sum;
}

__global__ void curandSetup_kernel(curandState *dev_curandState,
		unsigned long long seed) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, id, 0, &dev_curandState[id]);
}

__global__ void sample_kernel(double *dev_prob, int *dev_samples,
		curandState *dev_curandState, int ordersNum) {
	extern __shared__ double prob[];
	prob[threadIdx.x] = dev_prob[threadIdx.x];
	__syncthreads();

	curandState localState = dev_curandState[threadIdx.x];
	double r = curand_uniform_double(&localState);
	dev_curandState[threadIdx.x] = localState;

	dev_samples[threadIdx.x] = binarySearch(prob, ordersNum, r);
	dev_samples[threadIdx.x] = threadIdx.x;
}

__global__ void calcOnePairPerThread_kernel(double * dev_lsTable,
		int * dev_newOrders, double * dev_parentSetScore, int nodesNum,
		int allParentSetNumPerNode, int parentSetNumInOrder) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = ((gridDim.y * blockDim.x) * idy) + idx;
	int orderId = id / parentSetNumInOrder;
	int parentSetIndex = id % parentSetNumInOrder + 1;
	int nodePos, parentSetSize;
	int parentSet[CONSTRAINTS], i = 0;

	// find nodePos and parentSetSize
	for (nodePos = 0; nodePos < nodesNum; nodePos++) {
		for (parentSetSize = 0;
				parentSetSize <= CONSTRAINTS && parentSetSize < nodePos + 1;
				parentSetSize++) {
			parentSetIndex = parentSetIndex - C_kernel(parentSetSize, nodePos);
			if (parentSetIndex <= 0) {
				parentSetIndex = parentSetIndex
						+ C_kernel(parentSetSize, nodePos);
				i = 1;
				break;
			}
		}
		if (i == 1) {
			break;
		}
	}

	// find combination
	findCombWithSize_kernel(nodePos + 1, parentSetIndex, parentSetSize,
			parentSet);

	//find parentSet
	int curNode = dev_newOrders[orderId * nodesNum + nodePos];
	for (i = 0; i < parentSetSize; i++) {
		parentSet[i] = dev_newOrders[orderId * nodesNum + parentSet[i] - 1];
		if (parentSet[i] > curNode) {
			parentSet[i] -= 1;
		}
	}

	//sort parentSet
	sortArray_kernel(parentSet, parentSetSize);

	//find index
	i = 0;
	if (parentSetSize > 0) {
		i = findIndex_kernel(nodesNum, parentSetSize, parentSet);
	}

	dev_parentSetScore[id] = dev_lsTable[(curNode - 1) * allParentSetNumPerNode + i];
}

__global__ void calcMaxParentSetScoreForEachNode_kernel(
		double *dev_parentSetScore, double *dev_maxLocalScore,
		int parentSetNumInOrder, int nodesNum) {
	int parentSetStartPos = 0;
	int i, j;
	for (i = 0; i < blockIdx.x; i++) {
		for (j = 0; j <= CONSTRAINTS && j < i + 1; j++) {
			parentSetStartPos += C_kernel(j, i);
		}
	}
	int parentSetEndPos = parentSetStartPos;
	for (i = 0; i <= CONSTRAINTS && i < blockIdx.x + 1; i++) {
		parentSetEndPos += C_kernel(i, blockIdx.x);
	}

	double max = dev_parentSetScore[threadIdx.x * parentSetNumInOrder
			+ parentSetStartPos];
	for (i = parentSetStartPos + 1; i < parentSetEndPos; i++) {
		double cur = dev_parentSetScore[threadIdx.x * parentSetNumInOrder + i];
		if (cur > max) {
			max = cur;
		}
	}
	dev_maxLocalScore[threadIdx.x * nodesNum + blockIdx.x] = max;
}
