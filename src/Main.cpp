#include "BNSL_GPU.cuh"
#include "cpu_function.h"
#include <stdlib.h>

int main() 
{
	calcCPUTimeStart("init: ");
	BNSL_init();
	calcCPUTimeEnd();

	calcCPUTimeStart("calcLs: ");
	BNSL_calcLocalScore();
	calcCPUTimeEnd();
	
	calcCPUTimeStart("start: ");
	BNSL_start();
	calcCPUTimeEnd();
	
	BNSL_printResult();

	calcCPUTimeStart("finish: ");
	BNSL_finish();
	calcCPUTimeEnd();

	return 0;
}
