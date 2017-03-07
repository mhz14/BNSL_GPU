#include "hp.h"

mpf_t sum;
mpf_t e;
mpz_t power;
mpf_t *temp;
int size;

void initGMP(int n){
	mpf_init(sum);
	mpf_init(e);
	mpz_init(power);
	mpf_set_d(e, 2.71828);
	temp = (mpf_t *)malloc(n * sizeof(mpf_t));
	for (int i = 0; i < n; i++){
		mpf_init(temp[i]);
	}
	size = n;
}

void normalize(double *prob){

	mpf_set_d(sum, 0);
	int i;
	for (i = 0; i < size; i++){
		mpz_set_d(power, prob[i]);
		mpf_pow_ui(temp[i], e, mpz_get_ui(power));
		mpf_add(sum, sum, temp[i]);
	}

	for (i = 0; i < size; i++){
		mpf_div(temp[i], temp[i], sum);
		prob[i] = mpf_get_d(temp[i]);
	}
}

void finishGMP(){
	mpf_clear(sum);
	mpf_clear(e);
	mpz_clear(power);
	for (int i = 0; i < size; i++){
		mpf_clear(temp[i]);
	}
	free(temp);
}
