#ifndef GPU_MODEL_CUDA_NOSTREAM_INIT_CUH
#define GPU_MODEL_CUDA_NOSTREAM_INIT_CUH
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include "utils.hpp"
#include "bert_state.hpp"

typedef struct _gpu_cuda_context_t {
	int num_streams;

	cudaStream_t *streams;
	cublasHandle_t *cublas_handles;
//	cusparseHandle_t *cusparse_handles;

	/// host memory (Col-major)
//	float *h_onevec;        // [1 x hidden_size]
//	float *h_onemat;        // [hidden_size x hidden_size]
	float **h_attention_mask;   // [batch][seq_len x seq_len]
	float **h_input;    // [batch][hidden x seq]
	float ***h_weight;  // [num_layer][weight_idx][...]
	float **h_weight_attention_fc_splitted;    // [num_layer][num_heads x head_size x hidden_size]
	float **h_weight_attention_fc_bias_splitted;    // [num_layer][seq_len x hidden_size]
	float **h_weight_ffw_prev;                  // [num_layer][hidden_size x feedforward_size]
	float **h_weight_ffw_prev_bias;            // [num_layer][seq_len x feedforward_size]
	float **h_weight_ffw_post_splitted;        // [num_layer][num_heads x ffw_chunk_size x hidden_size]
	float **h_weight_ffw_post_bias_splitted;   // [num_layer][seq_len x hidden_size]
	float **h_buf_query;   // [batch][seq_len x hidden_size]
	float **h_buf_key;     // [batch][seq_len x hidden_size]
	float **h_buf_value;   // [batch][seq_len x hidden_size]
	float **h_buf_score;   // [batch][num_heads x seq_len x seq_len]
	float **h_buf_expsum;  // [batch][num_heads x seq_len x 1]
	float **h_buf_softmax; // [batch][num_heads x seq_len x seq_len]
	float **h_buf_att;      // [batch][seq_len x hidden_size]
	float **h_buf_att_fc_result_split; // [batch][num_heads x seq_len x hidden_size]
	float **h_buf_att_layernorm;    // [batch][seq_len x head_size]
	float **h_buf_ffw_intermediate;    // [batch][seq_len x feedforward_size]
	float **h_buf_ffw_gelu;            // [batch][seq_len x feedforward_size]
	float **h_buf_ffw_result_split;    // [batch][num_heads x seq_len x hidden_size]
	float **h_buf_ffw_layernorm;    // [batch][seq_len x hidden_size]

//	float **h_buf_layernorm_mean;   // [batch][seq_len x hidden_size]
//	float **h_buf_layernorm_tmp;    // [batch][seq_len x hidden_size]
	float **h_buf_layernorm_nrm_v;  // [batch][seq_len x 1]

	/// GPU memory
//	float *d_onevec;        // [1 x hidden_size]
//	float *d_onemat;        // [hidden_size x hidden_size]
	float **d_attention_mask;   // [batch][seq_len x seq_len]
	float **d_input;        // [batch][hidden x seq]
	float ***d_weight;      // [num_layer][weight_idx][...]
	float **d_weight_attention_fc_splitted;    // [num_layer][num_heads x head_size x hidden_size]
	float **d_weight_attention_fc_bias_splitted;    // [num_layer][seq_len x hidden_size]
	float **d_weight_ffw_prev;                  // [num_layer][hidden_size x feedforward_size]
	float **d_weight_ffw_prev_bias;             // [num_layer][seq_len x feedforward_size]
	float **d_weight_ffw_post_splitted;        // [num_layer][num_heads x ffw_chunk_size x hidden_size]
	float **d_weight_ffw_post_bias_splitted;    // [num_layer][seq_len x hidden_size]
	float **d_buf_query;    // [batch][seq_len x hidden_size]
	float **d_buf_key;     // [batch][seq_len x hidden_size]
	float **d_buf_value;   // [batch][seq_len x hidden_size]
	float **d_buf_score;   // [batch][num_heads x seq_len x seq_len]
	float **d_buf_expsum;  // [batch][num_heads x seq_len x 1]
	float **d_buf_softmax; // [batch][num_heads x seq_len x seq_len]
	float **d_buf_att;      // [batch][seq_len x hidden_size]
	float **d_buf_att_fc_result_split; // [batch][num_heads x seq_len x hidden_size]
	float **d_buf_att_layernorm;    // [batch][seq_len x hidden_size]
	float **d_buf_ffw_intermediate;    // [batch][seq_len x feedforward_size]
	float **d_buf_ffw_gelu;            // [batch][seq_len x feedforward_size]
	float **d_buf_ffw_result_split;    // [batch][num_heads x seq_len x hidden_size]
	float **d_buf_ffw_layernorm;    // [batch][seq_len x hidden_size]

//	float **d_buf_layernorm_mean;   // [batch][seq_len x hidden_size]
//	float **d_buf_layernorm_tmp;    // [batch][hidden_size x seq_len]
	float **d_buf_layernorm_nrm_v;  // [batch][seq_len x 1]

} gpu_cuda_context_t;

int cuda_init(Params *params, BERT_State *bert_state, gpu_cuda_context_t *gpu_context);
int cuda_mem_alloc(Params *params, BERT_State *bert_state, gpu_cuda_context_t *gpu_context);
int cuda_mem_init(Params *params, BERT_State *bert_state, gpu_cuda_context_t *gpu_context);
int cuda_bert_main(Params *params, BERT_State *bert_state, gpu_cuda_context_t *gpu_context);
void sync_all_buf_to_host(BERT_State *bert_state, gpu_cuda_context_t *gpu_context);
void cuda_context_deinit(BERT_State *bert_state, gpu_cuda_context_t *gpu_context);
#endif //GPU_MODEL_CUDA_NOSTREAM_INIT_CUH
