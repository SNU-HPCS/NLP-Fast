#include <cstdio>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include <mkl.h>

#include "log.hpp"

static float *rs_forward_one = NULL;

static float rs_forward(float *a, MKL_INT size) {
	float sum;

	sum = cblas_sdot(size, a, 1, rs_forward_one, 1);
	return sum;
}

struct sm_arg {
	float **a;
	float **output;
	MKL_INT m;
	MKL_INT k;
	MKL_INT rs_size;
	int num_heads;
};

//static void softmax_forward(float **a, float** output, MKL_INT m, MKL_INT k, MKL_INT rs_size, int num_heads) {
static void *softmax_forward(void *void_arg) {
	struct sm_arg *args = (struct sm_arg*)void_arg;
	float **a = args->a;
	float **output = args->output;
	MKL_INT m = args->m;
	MKL_INT k = args->k;
	MKL_INT rs_size = args->rs_size;
	int num_heads = args->num_heads;

	int id = 0;
	float sum;
	int test_size = 256;
	float *test_in, *test_out;

	test_in = (float*)mkl_malloc(test_size * sizeof(float), 64);
	test_out = (float*)mkl_malloc(test_size * sizeof(float), 64);
	vsExp(test_size, test_in, test_out);

	//printf("gnu exp\n");
	//for(int i=0; i<num_heads; i++){
		//for (int j = 0; j < m*k; j++) {
			//output[i][j] = exp(a[i][j]);
			//__asm__ __volatile__ ( "vzeroupper" : : : );
		//}
	//}

	int steps = 8;
	int entries = m*k/steps;
	printf("m:%d, k:%d, step:%d entries:%d\n", m, k, steps, entries);

	printf("mkl exp\n");
	for(int i=0; i<num_heads; i++){
		for (int j=0; j<steps; j++) {
			vsExp(entries, a[i]+entries*j, output[i]+entries*j);
		}
	}

	//sum
	for(int i=0; i<num_heads;i++){
		for(int j=0; j<m; j++){
			sum = rs_forward(&output[i][k*j], rs_size) + 1;
			cblas_sscal(k, (1/sum), &output[i][k*j], 1);
		}
	}
}

int main(int argc, char *argv[]) {
	MKL_INT rs_size = 64;
//	int num_heads = 16;
	int num_heads = 4;
	MKL_INT m = 256;
	MKL_INT k = 256;
	float **a = NULL;
	float **output = NULL;

	// init various variables
	rs_forward_one = (float*)mkl_calloc(rs_size, sizeof(float), 64);
	for (int i = 0; i < rs_size; i++) {rs_forward_one[i] = 1;}

	a = (float**)malloc(sizeof(float*) * num_heads);
	output = (float**)malloc(sizeof(float*) * num_heads);
	for (int i = 0; i < num_heads; i++) {
		a[i] = (float*)mkl_calloc(m*k, sizeof(float), 64);
		memset(a[i], 0, m*k*sizeof(float));
		output[i] = (float*)mkl_calloc(m*k, sizeof(float), 64);
		memset(output[i], 0, m*k*sizeof(float));
	}

	mkl_set_num_threads(1);
	struct sm_arg _sm_arg = {
			.a = a,
			.output = output,
			.m = m,
			.k = k,
			.rs_size = rs_size,
			.num_heads = num_heads
	};
	// Do calculation
	//{
		//pthread_t t_thread;
		//pthread_create(&t_thread, NULL, softmax_forward, &_sm_arg);
		//pthread_join(t_thread, NULL);
	//}
	softmax_forward(&_sm_arg);
}
