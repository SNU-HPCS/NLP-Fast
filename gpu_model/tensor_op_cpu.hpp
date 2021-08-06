#ifndef MODEL_TENSOR_OP_CPU_HPP
#define MODEL_TENSOR_OP_CPU_HPP
#include <mkl.h>


void gather(int *a, float *table, MKL_INT seq, MKL_INT size, float* output);
void add(float *a, float *b, MKL_INT size, float *output);
void one_hot(MKL_INT m, MKL_INT k, int * id, float *output);
void squeeze(float **in, int batch, MKL_INT k, float *output);

class matmul{
private:
	CBLAS_LAYOUT layout = CblasRowMajor;
	CBLAS_TRANSPOSE transA = CblasNoTrans;
	CBLAS_TRANSPOSE transB = CblasNoTrans;
	float alpha = 1.0;
	float beta = 0.0;
	MKL_INT m, n, k;
	MKL_INT lda, ldb, ldc;

public:
	matmul(MKL_INT _m, MKL_INT _n, MKL_INT _k);
	void forward(float *a, float *b, bool is_trans, float* output);
};

class bias_add{// done without train
private:
	MKL_INT m;
	MKL_INT k;
	bool is_training;

public:
	float *bias;

	bias_add() {}
	bias_add(bool _is_training, float *_bias, MKL_INT _m, MKL_INT _k);
	void forward(float *a);
};

class layer_dense{
private:
	float alpha = 1.0;
	float beta = 0.0;

	bool is_training;
	MKL_INT m, n, k; //weight matrix size(k x n), input size(m x k)
	MKL_INT lda, ldb, ldc;
	CBLAS_LAYOUT layout = CblasRowMajor;
	CBLAS_TRANSPOSE transA = CblasNoTrans;

public:
	float *weight;
	bias_add b_add;

	layer_dense(float *_weight, float *_bias, bool _is_training, MKL_INT _m, MKL_INT _n, MKL_INT _k);
	void forward(float *a, CBLAS_TRANSPOSE transB, float *output);
};

class reduce_mean{//done without train
private:
	float *x;
	MKL_INT size;
	float *one;
public:
	reduce_mean(){}
	reduce_mean(MKL_INT _size);
	float forward(float *a);
	void reduce_mean_deinit();
};

class layer_norm{
private:
	float *gamma;
	float *beta;

	int batch;
	MKL_INT m;
	MKL_INT k;

	reduce_mean rm;

	float **temp1;

	void vector_scalar_sub(float *src, float *dest, float s, MKL_INT size);
public:
	layer_norm() {}
	layer_norm(float* _gamma, float* _beta, int _batch, MKL_INT _m, MKL_INT _k);
	void forward(float *a, int i, float *output);
	void layer_norm_deinit();
};
#endif //MODEL_TENSOR_OP_CPU_HPP
