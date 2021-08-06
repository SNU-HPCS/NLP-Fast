#include <chrono>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <mkl.h>
#include <cstring>
#include <cassert>

#include "tensor_func.hpp"
#include "log.hpp"

using namespace std;
using std::chrono::system_clock;

//for debugging//
void print_t(float**p, int batch, int row, int col) {
	int count=0;
    printf("\n");
	for(int i=0; i<batch; i++){
		for(int j=0; j<row*col; j++){
			std::cout<<p[i][j]<<"  ";
			count++;
			if(count == col){
				count = 0;
				std::cout<<"\n";
			}
		}
		std::cout<<"\n";
	}
}
void print_ti(int**p, int batch, int row, int col) {
	int count=0;
	for(int i=0; i<batch; i++){
		for(int j=0; j<row*col; j++){
			std::cout<<p[i][j]<<"  ";
			count++;
			if(count == col){
				count = 0;
				std::cout<<"\n";
			}
		}
		std::cout<<"\n";
	}
}

void squeeze(float **in, int batch, MKL_INT k, float *output){
	for(int i=0; i<batch; i++){
		cblas_scopy(k, in[i], 1, (((float*)output)+i*k), 1);
	}
}


void gather(int *a, float *table, MKL_INT seq, MKL_INT hidden_size, float *output){
	for(int j=0; j<seq; j++){
		cblas_scopy(hidden_size, &table[hidden_size*a[j]], 1, &output[j*hidden_size], 1);
	}
}

void split(float *input, MKL_INT m, MKL_INT k, int num_heads, float **output) {
	MKL_INT head_size;
	head_size = k/num_heads;
    for(int i=0; i<num_heads; i++){
        for(int j=0; j<m; j++){
            cblas_scopy(head_size, (input + k*j + head_size*i), 1, (output[i] + head_size*j), 1); 
        }
    }
}

void merge(float **a,  MKL_INT m, MKL_INT k, int num_heads, float* output) {
	MKL_INT head_size = k/num_heads;
	for(int j=0; j<m; j++){
		for(int l=0; l<num_heads; l++){
			cblas_scopy(head_size, (a[l]+j*head_size), 1, (output + j*k + l*head_size), 1);
		}
	}
}

float ** ones(int batch, MKL_INT m, MKL_INT k) { //done

	float **ret=(float **)mkl_malloc(batch * sizeof(float *), 64);
	for(int i=0;i<batch; i++){
		ret[i]=(float*)mkl_malloc(m*k*sizeof(float), 64);
	}
	for(int i=0;i<batch;i++){
		for(int j=0; j<m*k; j++){
			ret[i][j]=1;
		}
	}
	return ret;
}

void ones(MKL_INT m, MKL_INT k, float* output){
	for(int i=0; i<m*k; i++)
		output[i]=1;
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

reduce_mean::reduce_mean(bool _is_training, MKL_INT _size) {
	size = _size;
	is_training = _is_training;
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

reduce_sum::reduce_sum(bool _is_training, MKL_INT _size) {
	size = _size;
	is_training = _is_training;
	one = (float*)mkl_calloc(size, sizeof(float), 64);
	for(int i=0; i<size; i++){
		one[i]=1;
	}
}

float reduce_sum::forward(float *a) {
	return cblas_sdot(size, a, 1, one, 1);
}

void reduce_sum::reduce_sum_deinit() {
	mkl_free(one); one = nullptr;
}

gelu::gelu(bool _is_training, int _batch, MKL_INT _size) {
	batch = _batch;
	size = _size;
	one_mat = (float*)mkl_calloc(size, sizeof(float), 64);
	for(int i=0; i<size; i++){
		one_mat[i] = 1;
	}
	temp = (float**)mkl_calloc(batch, sizeof(float*), 64);
	temp1 = (float**)mkl_calloc(batch, sizeof(float*), 64);
	
	for(int i=0; i<batch; i++){
		temp[i] = (float*)mkl_calloc(size, sizeof(float), 64);
		temp1[i] = (float*)mkl_calloc(size, sizeof(float), 64);
		memset(temp[i], 0, sizeof(float) * size);
		memset(temp1[i], 0, sizeof(float) * size);
	}
}

void gelu::forward(float *a, int id, float *output) {
	vsMul(size, a, a, temp[id]);               // temp <= a^2
	vsMul(size, a, temp[id], temp1[id]);    // temp1 <= a^3
	cblas_sscal(size, 0.044715, temp1[id], 1);  // temp1 <= 0.044715 * a^3
	vsAdd(size, temp1[id], a, temp[id]);   // temp <= a + 0.044715 * a^3
	cblas_sscal(size, sqrt(2/M_PI), temp[id], 1);  // temp <= root(2/PI) * (a + 0.044715 * a^3)
	vsTanh(size, temp[id], temp1[id]);     // temp1 <= tanh(root(2/PI) * (a + 0.044715 * a^3))
	vsAdd(size, temp1[id], one_mat, temp[id]); // temp <= tanh(root(2/PI) * (a + 0.044715 * a^3)) + 1
	cblas_sscal(size, 0.5, temp[id], 1);   // temp <= (tanh(root(2/PI) * (a + 0.044715 * a^3)) + 1) / 2
	vsMul(size, temp[id], a, output);  // output <= (tanh(root(2/PI) * (a + 0.044715 * a^3)) + 1) / 2 * a
}

void gelu::gelu_deinit() {
	mkl_free(one_mat); one_mat = nullptr;

	for(int i=0; i<batch; i++){
		mkl_free(temp[i]);
		mkl_free(temp1[i]);
	}
	mkl_free(temp); temp = nullptr;
	mkl_free(temp1); temp1 = nullptr;
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

//void normalize::forward(float **a, int num_heads, MKL_INT size, float factor, bool is_sqrt) {//size inicates total m*k
//	if(is_sqrt){
//		normalize_factor = sqrt(factor);
//	}
//	else{
//		normalize_factor = factor;
//	}
//	for(int i=0;i<num_heads; i++){
//		cblas_sscal(size, normalize_factor, a[i], 1);
//	}
//}

void normalize::forward(float *a, MKL_INT size, float factor, bool is_sqrt) {//size inicates total m*k
	if(is_sqrt){
		normalize_factor = sqrt(factor);
	}
	else{
		normalize_factor = factor;
	}
	cblas_sscal(size, normalize_factor, a, 1);
}

softmax::softmax(bool _is_training, MKL_INT _m, MKL_INT _k) {
	is_training = _is_training;
	m = _m;
	k = _k;
	rs = reduce_sum(is_training, k);
}

void softmax::forward(float *a, float *output) {
	float sum;

	// exp
	vsExp(m*k, a, output);

	//sum & div
	for(int i = 0; i < m; i++){
		sum = rs.forward(&output[k*i]);
		cblas_sscal(k, (1/sum), &output[k * i], 1);
	}
}

void softmax::softmax_deinit() {
	rs.reduce_sum_deinit();
}

partial_softmax::partial_softmax(bool _is_training, MKL_INT _m, MKL_INT _k) {
	is_training = _is_training;
	m = _m;
	k = _k;
	rs = reduce_sum(is_training, k);
}

void partial_softmax::forward(float *a, float *output, float *partial_sum) {
	// exp
	vsExp(m*k, a, output);

	// partial sum
	for(int i = 0; i < m; i++){
		partial_sum[i] += rs.forward(&output[k*i]);
	}
}

void partial_softmax::partial_softmax_deinit() {
	rs.reduce_sum_deinit();
}

layer_norm::layer_norm(float *_gamma, float *_beta, bool _is_training, int _batch, MKL_INT _m, MKL_INT _k) {
	is_training = _is_training;
	batch = _batch;
	m = _m;
	k = _k;
	gamma = _gamma;
	beta = _beta;

	rm = reduce_mean(is_training, k);

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

matmul::matmul(bool _is_training, MKL_INT _m, MKL_INT _n, MKL_INT _k){
	is_training = _is_training;
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

void matmul::forward(float **a, float **b, bool is_trans, float **output, int batch) {
	if(is_trans){
		transB = CblasTrans;
		ldb = k;
	}
	for(int i=0; i<batch; i++){
		cblas_sgemm(layout, transA , transB, m, n, k, alpha, a[i], lda, b[i], ldb, beta, output[i], ldc);
	}
}