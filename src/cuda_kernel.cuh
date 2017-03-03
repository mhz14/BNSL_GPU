#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include "config.h"
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

// 检查CUDA错误信息
void CheckCudaError(cudaError_t err, char *errMsg);
#define CUDA_CHECK_RETURN(value1, value2) CheckCudaError(value1, value2)

// 统计GPU运行时间
void calcGPUTimeStart(char *message);
void calcGPUTimeEnd();

// ----device kernel----

// k-combination按字典序映射为整数值
// nodesNum 为结点数量
// k 为k-combination子序列的长度
// combination 为k-combination的值
// 返回映射的整数值
__device__ int findIndex_kernel(int nodesNum, int k, int *combination);

// 整数值按字典序映射为k-combination
// nodesNum 为结点数量
// index 为按字典序映射的整数值
// size 为返回的k-combination子序列的长度
// combination 为k-combination的值
__device__ void findComb_kernel(int nodesNum, int index, int *size, int *combination);

// 整数值按字典序映射为k-combination
// nodesNum 为结点数量
// index 为按字典序映射的整数值
// size 为已知的k-combination子序列的长度
// combination 为k-combination的值
__device__ void findCombWithSize_kernel(int nodesNum, int index, int size, int* combi);

// 组合变为父结点集合
// vi 当前结点
// combi 组合
// size 组合大小
__device__ void recoverComb_kernel(int vi, int *combi, int size);

// 返回m个数中选n个的组合数
__device__ long C_kernel(int n, int m);

// 对数组进行排序
// s 排序数组
// n 数组大小
__device__ void sortArray_kernel(int *s, int n);

// 给定结点id和父结点集合，计算局部得分
// dev_valuesRange 结点取值范围
// dev_samplesValues 样本取值
// samplesNum 样本数量
// parentSet 父结点集合
// size 父结点集合元素个数
// curNode 当前结点
// nodesNum 结点数量
__device__ double calLocalScore_kernel(int *dev_valuesRange, int *dev_samplesValues, int samplesNum, int *parentSet, int size, int curNode, int nodesNum);

// 二分查找概率表采样
// prob 概率分布
// ordersNum 拓扑排序数量
// r 概率值[0,1]
__device__ int binarySearch(double *prob, int ordersNum, double r);

// ----global kernel----

// 计算所有结点和父结点集合的局部得分
// dev_valuesRange 结点取值范围
// dev_samplesValues 样本取值
// dev_lsTable 局部得分表
// samplesNum 样本数量
// nodesNum 结点数量
// parentSetNum 需要计算的父结点集合的个数
__global__ void calcAllPossibleLocalScore_kernel(int *dev_valuesRange, int *dev_samplesValues, double *dev_lsTable, int samplesNum, int nodesNum, int parentSetNum);

// 随机生成N个拓扑排序
// dev_newOrders 生成的拓扑排序
// dev_curandState GPU中存储curand的随机状态
// nodesNum 结点数量
__global__ void generateOrders_kernel(int *dev_newOrders, curandState *dev_curandState, int nodesNum);

// 计算所有拓扑排序的得分
// dev_maxLocalScore 每个结点父结点集合的最高得分
// dev_ordersScore 拓扑排序的得分
// nodesNum 结点数量
__global__ void calcAllOrdersScore_kernel(double *dev_maxLocalScore, double *dev_ordersScore, int nodesNum);

// 初始化curand的状态
// dev_curandState GPU中存储curand的随机状态
// seed curand随机种子
__global__ void curandSetup_kernel(curandState *dev_curandState, unsigned long long seed);

// 对I进行采样
// dev_prob I的分布
// dev_samples 样本
// dev_curandState GPU中存储curand的随机状态
// ordersNum 拓扑排序的数量
__global__ void sample_kernel(double *dev_prob, int *dev_samples, curandState *dev_curandState, int ordersNum);

// 计算每一对结点和父结点集合的局部得分
// dev_lsTable 局部得分表
// dev_newOrders 生成的拓扑排序
// dev_parentSetScore 符合拓扑排序的每一对结点和父结点集合的得分
// nodesNum 结点数量
// parentSetNum 所有可能的父结点集合的个数
// parentSetNumInOrder 一个拓扑排序中可能的父结点集合的个数
__global__ void calcOnePairPerThread_kernel(double * dev_lsTable, int * dev_newOrders, double * dev_parentSetScore, int nodesNum, int parentSetNum, int parentSetNumInOrder);

// 计算每个结点父结点集合的最高得分
// dev_parentSetScore 符合拓扑排序的每一对结点和父结点集合的得分
// dev_maxLocalScore 每个结点父结点集合的最高得分
// parentSetNumInOrder 一个拓扑排序中可能的父结点集合的个数
// nodesNum 结点数量
__global__ void calcMaxParentSetScoreForEachNode_kernel(double *dev_parentSetScore, double *dev_maxLocalScore, int parentSetNumInOrder, int nodesNum);