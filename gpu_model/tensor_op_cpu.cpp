#include <iostream>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <mkl.h>

#include "tensor_op_cpu.hpp"


void gather(int *a, float *table, MKL_INT seq, MKL_INT hidden_size, float *output){
	for(int j=0; j<seq; j++){
		cblas_scopy(hidden_size, &table[hidden_size*a[j]], 1, &output[j*hidden_size], 1);
	}
}

void add(float *a, float *b, MKL_INT size, float *output) {
	vsAdd(size, a, b, output);
}

void one_hot(MKL_INT m, MKL_INT k, int * id, float* output) {
	for(int j=0; j<m; j++){
		if(id[j]>=k){
			std::cout<<"one_hot: invalid id"<<std::endl;
		}
		output[j*k + id[j]]=1;
	}
}

void squeeze(float **in, int batch, MKL_INT k, float *output){
	for(int i=0; i<batch; i++){
		cblas_scopy(k, in[i], 1, (((float*)output)+i*k), 1);
	}
}

matmul::matmul(MKL_INT _m, MKL_INT _n, MKL_INT _k){
	m = _m;
	n = _n;
	k = _k;
	lda = _k;
	ldb = _n;
	ldc = _n;
}

void matmul::forward(float *a, float *b, bool is_trans, float *output){
	if(is_trans){
		transB = CblasTrans;
		ldb = k;
	}
	cblas_sgemm(layout, transA , transB, m, n, k, alpha, a, lda, b, ldb, beta, output, ldc);
}

bias_add::bias_add(bool _is_training, float *_bias, MKL_INT _m, MKL_INT _k) {
	bias = _bias;
	is_training = _is_training;
	m = _m;
	k = _k;
}

void bias_add::forward(float *a) {
	for(int i=0; i<m; i++){
		cblas_saxpy(k, 1, bias, 1, (a+i*k), 1);
	}
}

layer_dense::layer_dense(float *_weight, float *_bias, bool _is_training, MKL_INT _m, MKL_INT _n, MKL_INT _k) {
	//set parameter
	m = _m;
	n = _n;
	k = _k;
	lda = _k;
	ldb = _n;
	ldc = _n;
	is_training = _is_training;
	//make weight matrix and bias
	b_add = bias_add(is_training, _bias, m, n);
	weight = _weight;
	//make output
}
void layer_dense::forward(float *a, CBLAS_TRANSPOSE transB, float *output) {
	cblas_sgemm(layout, transA, transB, m, n, k, alpha, a, lda, weight, ldb, beta, output, ldc);
	b_add.forward(output);
}

reduce_mean::reduce_mean(MKL_INT _size) {
	size = _size;
	one = (float*)mkl_calloc(size, sizeof(float), 64);
	for(int i=0; i<size; i++){
		one[i]=1;
	}
}

float reduce_mean::forward(float *a) {
	return (float) cblas_sdot(size, a, 1, one, 1) / (float) size;
}

void reduce_mean::reduce_mean_deinit() {
	mkl_free(one); one = nullptr;
}

layer_norm::layer_norm(float *_gamma, float *_beta, int _batch, MKL_INT _m, MKL_INT _k) {
	batch = _batch;
	m = _m;
	k = _k;
	gamma = _gamma;
	beta = _beta;

	rm = reduce_mean(k);

	temp1 = (float**)mkl_calloc(batch, sizeof(float*), 64);
	for(int i=0; i<batch; i++){
		temp1[i] = (float*)mkl_calloc(m, k * sizeof(float), 64);
		memset(temp1[i], 0, sizeof(float) * m * k);
	}
}
void layer_norm::forward(float *a, int id, float *output) {
	float mean;
	float var_;
	float uc_norm;
	for(int j=0; j<m; j++){
		mean = rm.forward(&a[k*j]);
		vector_scalar_sub((a+k*j), (temp1[id]+k*j), mean, k);
		uc_norm = cblas_snrm2(k, &temp1[id][k*j], 1);
		var_ = uc_norm*uc_norm / (float)k + FLT_EPSILON;
		cblas_sscal(k,1/(sqrt(var_)), &temp1[id][k*j], 1);
		vsMul(k, gamma, &temp1[id][k*j], &output[k*j]);
		vsAdd(k, beta, &output[k*j], &output[k*j]);
	}
}

void layer_norm::vector_scalar_sub(float *src, float *dest, float s, MKL_INT size) {
	for(int i=0; i<size; i++){
		dest[i] = src[i] - s;
	}
}

void layer_norm::layer_norm_deinit() {
	rm.reduce_mean_deinit();

	for(int i=0; i<batch; i++) {
		mkl_free(temp1[i]);
	}
	mkl_free(temp1); temp1 = nullptr;
}
