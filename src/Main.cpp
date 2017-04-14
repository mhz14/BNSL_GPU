#include "BNSL_GPU.cuh"
#include "cpu_function.h"
#include <stdlib.h>

int main() 
{
	printf("BNSL_init starts. \n");
	BNSL_init();

	printf("BNSL_calcLocalScore starts. \n");
	BNSL_calcLocalScore();
	
	printf("BNSL_search starts. \n");
	BNSL_search();
	
	printf("BNSL_printResult starts. \n");
	BNSL_printResult();

	BNSL_finish();

	return 0;
}
