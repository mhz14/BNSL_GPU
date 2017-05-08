#include "cpu_function.h"

int len;

clock_t watch;

void startWatch(){
	watch = clock();
}

int stopWatch(){
	return (clock() - watch) / 1000;
}

void readNodeInfo(int *nodesNum, int **valuesRange){
	FILE * inFile = fopen(NODEINFO_PATH, "r");

	int count = 0;
	char cur = fgetc(inFile);
	while (cur != EOF){
		if (cur == '\n')
			count++;
		cur = fgetc(inFile);
	}
	*nodesNum = ++count;

	rewind(inFile);
	int *pointer = (int *)malloc(sizeof(int) * count);
	int i;
	for (i = 0; i < count; i++){
		fscanf(inFile, "%d", &(pointer[i]));
	}

	fclose(inFile);
	*valuesRange = pointer;
}

void readSamples(int **samplesValues, int *samplesNum, int nodesNum){
	FILE * inFile = fopen(SAMPLES_PATH, "r");
	int i, j, value;

	int count = 0;
	char cur = fgetc(inFile);
	while (cur != EOF){
		if (cur == '\n')
			count++;
		cur = fgetc(inFile);
	}
	*samplesNum = ++count;

	*samplesNum = SAMPLES_NUM;

	int *pointer = (int *)malloc(sizeof(int) * count * nodesNum);
	rewind(inFile);
	for (i = 0; i < count; i++){
		for (j = 0; j < nodesNum; j++){
			fscanf(inFile, "%d", &value);
			pointer[i*nodesNum + j] = value - 1;
		}
	}

	fclose(inFile);
	*samplesValues = pointer;
}

long C(int n, int m){

	if (n > m || n < 0 || m < 0)
		return -1;

	int k, res = 1;
	for (k = 1; k <= n; k++){
		res = (res*(m - n + k)) / k;
	}
	return res;
}

void randInitOrder(int *s, int nodesNum){
	for (int i = 0; i < nodesNum; i++){
		s[i] = i + 1;
	}
	int swap, r;
	srand((unsigned int)time(NULL));
	for (int i = nodesNum - 1; i > 0; i--){
		r = rand() % i;
		swap = s[r];
		s[r] = s[i];
		s[i] = swap;
	}
}

void selectTwoNodeToSwap(int *n1, int *n2, int nodesNum) {
	*n1 = rand() % nodesNum;
	*n2 = rand() % nodesNum;
	if (*n1 == *n2) {
		*n2 = rand() % (nodesNum - 1);
		if (*n2 >= *n1) {
			*n2++;
		}
	}
}

void randSwapTwoNode(int *order, int nodesNum) {
	int n1 = 0, n2 = 0, temp;
	selectTwoNodeToSwap(&n1, &n2, nodesNum);
	temp = order[n1];
	order[n1] = order[n2];
	order[n2] = temp;
}

int compare(const void*a, const void*b) {
	return *(int*) a - *(int*) b;
}

int calcValuesMaxNum(int *valuesRange, int nodesNum) {
	int * valuesRangeToSort = (int *) malloc(nodesNum * sizeof(int));
	memcpy(valuesRangeToSort, valuesRange, nodesNum * sizeof(int));
	qsort(valuesRangeToSort, nodesNum, sizeof(int), compare);
	int valuesMaxNum = 1;
	for (int i = nodesNum - CONSTRAINTS - 1; i < nodesNum; i++) {
		valuesMaxNum *= valuesRangeToSort[i];
	}
	free(valuesRangeToSort);
	return valuesMaxNum;
}

void calcCDFInit(int ordersNum){
	initGMP(ordersNum);
	len = ordersNum;
}

int calcCDF(double *ordersScore, double *prob){
	double min = 0.0, max = -DBL_MAX;
	int maxId = -1;
	int i;
	for (i = 0; i < len; i++){
		if (ordersScore[i] < min){
			min = ordersScore[i];
		}
		if (ordersScore[i] > max){
			max = ordersScore[i];
			maxId = i;
		}
	}

	for (i = 0; i < len; i++){
		prob[i] = ordersScore[i] - min;
	}

	normalize(prob);

	for (i = 1; i < len; i++){
		prob[i] = prob[i - 1] + prob[i];
	}

	return maxId;
}

void calcCDFFinish(){
	finishGMP();
}
