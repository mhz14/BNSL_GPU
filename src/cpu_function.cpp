#include "cpu_function.h"

int begin = 0;
int len = 0;

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

	int *pointer = (int *)malloc(sizeof(int) * count * nodesNum);
	rewind(inFile);
	for (i = 0; i < count; i++){
		for (j = 0; j < nodesNum; j++){
			fscanf(inFile, "%d", &value);
			pointer[i*nodesNum + j] = value;
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

void calcCPUTimeStart(char const *message){
	begin = clock();
	printf("%s", message);
}

void calcCPUTimeEnd(){
	printf("Elapsed CPU time is %dms\n", (clock() - begin) / 1000);
}
