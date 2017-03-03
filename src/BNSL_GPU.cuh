#include "cuda_kernel.cuh"
#include "cpu_function.h"

// 初始化
void BNSL_init();

// 计算所有局部得分
void BNSL_calLocalScore();

// 算法执行
void BNSL_start();

// 输出算法结果
void BNSL_printResult();

// 算法结束
void BNSL_finish();