#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

#include "tensor_op_gpu.cuh"

// Apply exp element-wisely to an array d_A
// d_A(m) = exp(d_A(n))
__global__ void g_exp(float *A, float *Out, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		Out[i] = expf(A[i]);
}


__global__ void g_exp_sum(float *inmat, float *outmat, float *outsumvec, int M_dim, int N_dim, int iters) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int Total_dim = M_dim * N_dim;

	if (i < M_dim * N_dim * iters) {
		int iter_idx = i / (Total_dim);
		float in_val = inmat[i];
		float exp_in_val = expf(in_val);
		outmat[i] = exp_in_val;
		atomicAdd(outsumvec + (iter_idx * M_dim) + (i % M_dim), exp_in_val);
	}
}


// Divide PQ by S for each question
// m: num_sentences
// n: num_questions
__global__ void g_normalize(float *_mat, const float *_vec, int M_dim, int N_dim, int iters) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int Total_dim = M_dim * N_dim;

	if (i < M_dim * N_dim * iters) {
		int iter_idx = i / Total_dim;
		_mat[i] /= _vec[(iter_idx * M_dim) + (i % M_dim)];
	}
}


__global__ void g_score_norm_layer_mask(float *_mat, float norm_factor, const float *attention_mask, int M_dim, int N_dim, int iters)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int Total_dim = M_dim * N_dim;

	if (i < M_dim * N_dim * iters) {
		float in_val = _mat[i];
		float mask_val = attention_mask[i % Total_dim];
		_mat[i] = in_val * norm_factor - mask_val;
	}
}

__global__ void g_layer_mean(float *inmat, float *outmeanvec, const int M_dim, const int N_dim) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < M_dim * N_dim) {
		float in_val = inmat[i];
		atomicAdd(outmeanvec + (i % M_dim), in_val / (float) N_dim);
	}
}
__global__ void g_layer_minus(float *inmat, float *invec, float *outmat, const int M_dim, const int N_dim) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < M_dim * N_dim) {
		float in_val = inmat[i];
		float in_vec_val = invec[i % M_dim];
		outmat[i] = in_val - in_vec_val;
	}
}

__global__ void g_layer_norm(float *inmat, float *outmat, float *tmpvec, float *gamma, float *beta, const int M_dim, const int N_dim) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	// cudaMemsetAsync(gpu_context->d_buf_layernorm_nrm_v[batch_idx], 0, M_dim * 1 * sizeof(float), gpu_context->streams[stream_idx]);
	if (i < M_dim) {
		tmpvec[i] = 0.0f;
	}
//	__syncthreads();

	// g_layer_mean
	if (i < M_dim * N_dim) {
		float in_val = inmat[i];
		atomicAdd(tmpvec + (i % M_dim), in_val / (float) N_dim);
	}
//	__syncthreads();

	// g_layer_minus
	if (i < M_dim * N_dim) {
		float in_val = inmat[i];
		float in_vec_val = tmpvec[i % M_dim];
		inmat[i] = in_val - in_vec_val;
	}

	/// Calculate norm2
	// cudaMemsetAsync
	if (i < M_dim) {
		tmpvec[i] = 0.0f;
	}
	this_thread_block().sync();
//	__syncthreads();

	// g_layer_snrm2
	if (i < M_dim * N_dim) {
		float in_val = inmat[i];
		atomicAdd(tmpvec + (i % M_dim), in_val * in_val);
	}
	this_thread_block().sync();
//	__syncthreads();

	// g_sqrt
	if (i < M_dim) {
		tmpvec[i] = sqrtf(tmpvec[i]);
	}
	this_thread_block().sync();
//	__syncthreads();

	// g_layer_norm_gamma_beta
	if (i < M_dim * N_dim) {
		float _var_reciprocal = 1.0f / (sqrtf(tmpvec[i % M_dim] * tmpvec[i % M_dim] / (float)N_dim + FLT_EPSILON));

		outmat[i] = inmat[i] * _var_reciprocal;
		outmat[i] = outmat[i] * gamma[i];
		outmat[i] = outmat[i] + beta[i];
	}
}

/**
 *
 * @param inmat: (M X N) matrix
 * @param outnrmvec: (M X 1) matrix
 * @param M_dim
 * @param N_dim
 */
__global__ void g_layer_snrm2(float *inmat, float *outnrmvec, const int M_dim, const int N_dim) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < M_dim * N_dim) {
		float in_val = inmat[i];
		atomicAdd(outnrmvec + (i % M_dim), in_val * in_val);
	}
}

__global__ void g_layer_norm_gamma_beta(float *inmat, float *outmat, float *nrmvec, float *gamma, float *beta, const int m, const int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < m * n) {
		float _var_reciprocal = 1.0f / (sqrtf(nrmvec[i % m] * nrmvec[i % m] / (float)n + FLT_EPSILON));

		outmat[i] = inmat[i] * _var_reciprocal;
		outmat[i] = outmat[i] * gamma[i];
		outmat[i] = outmat[i] + beta[i];
	}
}

__global__ void g_sqrt(float *a, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < N) {
		a[i] = sqrtf(a[i]);
	}
}

__global__ void g_gelu(float *In, float *Out, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < N) {
		const float scale = sqrtf(2.0f / CUDART_PI);
		float in_val = In[i];
		float cdf = 1.0f + tanhf(scale * (in_val + 0.044715f * (in_val * in_val * in_val)));
		cdf *= 0.5f;
		Out[i] = in_val * cdf;
	}
}