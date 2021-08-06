#ifndef GPU_MODEL_TENSOR_OP_CUH
#define GPU_MODEL_TENSOR_OP_CUH

__global__ void g_exp(float *A, float *Out, int N);
__global__ void g_exp_sum(float *inmat, float *outmat, float *outsumvec, int M_dim, int N_dim, int iters);
__global__ void g_normalize(float *PQ, const float *S, int M_dim, int N_dim, int iters);
__global__ void g_score_norm_layer_mask(float *_mat, float norm_factor, const float *attention_mask, int M_dim, int N_dim, int iters);
__global__ void g_layer_mean(float *inmat, float *outmeanvec, const int M_dim, const int N_dim);
__global__ void g_layer_minus(float *inmat, float *invec, float *outmat, const int M_dim, const int N_dim);
__global__ void g_layer_norm(float *inmat, float *outmat, float *tmpvec, float *gamma, float *beta, const int M_dim, const int N_dim);
__global__ void g_layer_snrm2(float *inmat, float *outnrmvec, const int M_dim, const int N_dim);
__global__ void g_layer_norm_gamma_beta(float *inmat, float *outmat, float *nrmvec, float *gamma, float *beta, const int m, const int n);
__global__ void g_sqrt(float *a, int N);
__global__ void g_gelu(float *In, float *Out, int N);

#endif //GPU_MODEL_TENSOR_OP_CUH
