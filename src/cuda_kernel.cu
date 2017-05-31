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

__device__ double calcLocalScore_kernel(int * valuesSize,
		int *dev_samplesValues, int *dev_N, int samplesNum, int* parentSet,
		int size, int curNode, int nodesNum, int valuesMaxNum) {

	int valuesNum = 1;
	for (int i = 0; i < size; i++) {
		valuesNum = valuesNum * valuesSize[parentSet[i] - 1];
	}

	int *N = dev_N + (blockIdx.x * blockDim.x + threadIdx.x) * valuesMaxNum;
	for (int i = 0; i < valuesMaxNum; i++) {
		N[i] = 0;
	}

	int pValueIndex = 0;
	for (int i = 0; i < samplesNum; i++) {
		pValueIndex = 0;
		for (int j = 0; j < size; j++) {
			pValueIndex = pValueIndex * valuesSize[parentSet[j] - 1]
					+ dev_samplesValues[i * nodesNum + parentSet[j] - 1];
		}
		N[pValueIndex * valuesSize[curNode]
				+ dev_samplesValues[i * nodesNum + curNode]]++;

	}

	double alpha = ALPHA / (valuesSize[curNode] * valuesNum);
	double localScore = size * log(GAMMA);
	for (int i = 0; i < valuesNum; i++) {
		int sum = 0;
		for (int j = 0; j < valuesSize[curNode]; j++) {
			int cur = i * valuesSize[curNode] + j;
			if (N[cur] != 0) {
				localScore = localScore + lgamma(N[cur] + alpha)
						- lgamma(alpha);
				sum = sum + N[cur];
			}
		}
		localScore = localScore + lgamma(alpha * valuesSize[curNode])
				- lgamma(alpha * valuesSize[curNode] + sum);
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
		int *dev_samplesValues, double *dev_lsTable, int *dev_N, int samplesNum,
		int nodesNum, int allParentSetNumPerNode, int valuesMaxNum) {

	extern __shared__ int valuesSize[];
	if (threadIdx.x < nodesNum) {
		valuesSize[threadIdx.x] = dev_valuesRange[threadIdx.x];
	}
	__syncthreads();

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < allParentSetNumPerNode) {
		int parentSet[CONSTRAINTS], parentSetSize = 0;
		findComb_kernel(nodesNum, id, &parentSetSize, parentSet);

		for (int curPos = 0; curPos < nodesNum; curPos++) {
			int parentSetTransferred[CONSTRAINTS];
			for (int i = 0; i < parentSetSize; i++) {
				parentSetTransferred[i] = parentSet[i];
			}
			recoverComb_kernel(curPos, parentSetTransferred, parentSetSize);
			dev_lsTable[curPos * allParentSetNumPerNode + id] =
					calcLocalScore_kernel(valuesSize, dev_samplesValues, dev_N,
							samplesNum, parentSetTransferred, parentSetSize,
							curPos, nodesNum, valuesMaxNum);
		}
	}
}

__global__ void generateOrders_kernel(int *dev_newOrders,
		curandState *dev_curandState, int nodesNum, int ordersNum) {
	extern __shared__ int oldOrder[];
	if (threadIdx.x < nodesNum) {
		oldOrder[threadIdx.x] = dev_newOrders[threadIdx.x];
	}
	__syncthreads();

	if (threadIdx.x < ordersNum) {
		for (int i = 0; i < nodesNum; i++) {
			dev_newOrders[threadIdx.x * nodesNum + i] = oldOrder[i];
		}

		if (threadIdx.x != 0) {
			curandState localState = dev_curandState[threadIdx.x];
			int i = threadIdx.x * nodesNum + curand(&localState) % nodesNum;
			int j = curand(&localState) % nodesNum;
			while (i == j) {
				j = curand(&localState) % nodesNum;
			}
			dev_curandState[threadIdx.x] = localState;
			j = threadIdx.x * nodesNum + j;
			int k = dev_newOrders[i];
			dev_newOrders[i] = dev_newOrders[j];
			dev_newOrders[j] = k;
		}
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

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < parentSetNumInOrder) {
		// find nodePos and parentSetSize
		int parentSetIndex = id + 1;
		int nodePos, parentSetSize;
		bool out = false;
		for (nodePos = 0; nodePos < nodesNum; nodePos++) {
			for (parentSetSize = 0;
					parentSetSize <= CONSTRAINTS && parentSetSize < nodePos + 1;
					parentSetSize++) {
				int curParentSetNum = C_kernel(parentSetSize, nodePos);
				if (parentSetIndex <= curParentSetNum) {
					out = true;
					break;
				}
				parentSetIndex -= curParentSetNum;
			}
			if (out) {
				break;
			}
		}

		// find combination
		int parentSet[CONSTRAINTS];
		findCombWithSize_kernel(nodePos + 1, parentSetIndex, parentSetSize,
				parentSet);

		//find transferred parentSet
		// orderId = blockIdx.y
		int curNode = dev_newOrders[blockIdx.y * nodesNum + nodePos];
		for (int i = 0; i < parentSetSize; i++) {
			parentSet[i] = dev_newOrders[blockIdx.y * nodesNum + parentSet[i]
					- 1];
			if (parentSet[i] > curNode) {
				parentSet[i] -= 1;
			}
		}

		//sort parentSet
		sortArray_kernel(parentSet, parentSetSize);

		//find index
		int i = 0;
		if (parentSetSize > 0) {
			i = findIndex_kernel(nodesNum, parentSetSize, parentSet);
		}

		dev_parentSetScore[blockIdx.y * parentSetNumInOrder + id] =
				dev_lsTable[(curNode - 1) * allParentSetNumPerNode + i];
	}
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
