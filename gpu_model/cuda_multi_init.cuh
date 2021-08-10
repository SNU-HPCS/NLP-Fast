#ifndef GPU_MODEL_CUDA_INIT_CUH
#define GPU_MODEL_CUDA_INIT_CUH
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include "utils.hpp"
#include "bert_state.hpp"

typedef struct _gpu_cuda_context_t {
	int num_streams;

	cudaStream_t *memcpy_streams;	// source streams (dedicated for each GPU)
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
	float **h_weight_ffw_prev;        // [num_layer][hidden_size x feedforward_size]
	float **h_weight_ffw_prev_bias;   // [num_layer][seq_len x feedforward_size]
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
	float **d_weight_ffw_prev;        // [num_layer][hidden_size x feedforward_size]
	float **d_weight_ffw_prev_bias;   // [num_layer][seq_len x feedforward_size]
	float **d_weight_ffw_post_splitted;        // [num_layer][num_heads x ffw_chunk_size x hidden_size]
	float **d_weight_ffw_post_bias_splitted;   // [num_layer][seq_len x hidden_size]
	float **d_buf_query;   // [batch][seq_len x hidden_size]
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

typedef struct {
	int gpu_num;
	int gpu_id;
	Params *params;
	BERT_State *bert_state;
	gpu_cuda_context_t *gpu_contexts;    // array of gpu_context

	pthread_barrier_t *multi_gpu_barrier_total_start;
	pthread_barrier_t *multi_gpu_barrier_local_att_fc_rsum;
	pthread_barrier_t *multi_gpu_barrier_local_att_fc_rsum_rescopy;
	pthread_barrier_t *multi_gpu_barrier_local_att_fc_rsum_rescopy_done;
	pthread_barrier_t *multi_gpu_barrier_local_ffw_rsum;
	pthread_barrier_t *multi_gpu_barrier_local_ffw_rsum_rescopy;
	pthread_barrier_t *multi_gpu_barrier_local_ffw_rsum_rescopy_done;
	pthread_barrier_t *multi_gpu_barrier_total_end;
} multi_gpu_thread_arg_t;

int get_device_count(int *num_gpus);
int set_device(int gpu_id);
int cuda_multi_cublas_init(Params *params, gpu_cuda_context_t *gpu_context, int gpu_id);
int cuda_multi_host_mem_alloc(Params *params, BERT_State *bert_state, gpu_cuda_context_t *gpu_context);
int cuda_multi_mem_alloc(Params *params, BERT_State *bert_state, gpu_cuda_context_t *gpu_context);
int cuda_multi_mem_init(Params *params, BERT_State *bert_state, gpu_cuda_context_t *gpu_context);
int cuda_multi_bert_main(multi_gpu_thread_arg_t* multi_gpu_arg);
void dump_gpu_matrix(float *gpu_mem, int M, int N, int gpu_id, const char *prefix_msg);
void sync_all_buf_to_host(BERT_State *bert_state, gpu_cuda_context_t *gpu_context);
void cuda_multi_host_context_deinit(BERT_State *bert_state, gpu_cuda_context_t *gpu_context, int num_gpus);
void cuda_multi_dev_context_deinit(BERT_State *bert_state, gpu_cuda_context_t *gpu_context);
#endif //GPU_MODEL_CUDA_INIT_CUH
