#include "config.h"
#include "mpf.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

// 读取结点个数与结点取值范围
void readNodeInfo(int *nodesNum, int **valuesRange);

// 读取结点观测数据
void readSamples(int **samplesValues, int *samplesNum, int nodesNum);

// 返回m个数中选n个的组合数
long C(int n, int m);

// 随机初始化一个拓扑排序
void randInitOrder(int *s, int nodesNum);

// 计算CDF累积概率分布
// ordersScore 拓扑排序的得分
// prob CDF累积概率分布
// ordersNum 拓扑排序的个数
int calcCDF(double *ordersScore, double *prob);

void calcCDFInit(int ordersNum);

void calcCDFFinish();

// 统计CPU运行时间
void calcCPUTimeStart(char *message);
void calcCPUTimeEnd();