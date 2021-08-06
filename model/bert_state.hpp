#ifndef MODEL_BERT_STATE_HPP
#define MODEL_BERT_STATE_HPP
#include <mkl.h>
#include "utils.hpp"
#include "log.hpp"
#include "tensor_func.hpp"

#define VALUE_TUNING_QKV_THRESHOLD      (5.0)
#define VALUE_TUNING_SCORE_THRESHOLD    (100.0)

typedef enum {
	EMB_WEIGHT_TABLE = 0,
	EMB_WEIGHT_TOKEN,
	EMB_WEIGHT_POSITION,
	EMB_WEIGHT_GAMMA,
	EMB_WEIGHT_BETA,
} EMB_WEIGHT_TYPE_ENUM_T;

typedef enum {
	WEIGHT_QLW = 0,
	WEIGHT_QLB,
	WEIGHT_KLW,
	WEIGHT_KLB,
	WEIGHT_VLW,
	WEIGHT_VLB,
	WEIGHT_ATTENTION,
	WEIGHT_ATTENTION_BIAS,
	WEIGHT_ATTENTION_GAMMA,
	WEIGHT_ATTENTION_BETA,
	WEIGHT_PREV_FFW,
	WEIGHT_PREV_FFB,
	WEIGHT_POST_FFW,
	WEIGHT_POST_FFB,
	WEIGHT_FF_GAMMA,
	WEIGHT_FF_BETA,
	WEIGHT_MAX_NUM,
} WEIGHT_TYPE_ENUM_T;

// Object for communication with attention threads
struct Attention_State {
	EXEC_MODE exec_mode;
	int batch_tid;
	int head_tid;

	int num_batchs;
	int batch_th_num;
	int num_heads;
	int head_th_num;
	int num_layers;
	bool column_based;
	int num_chunks;
	MKL_INT chunk_size;
	MKL_INT seq_length;
	MKL_INT hidden_size;
	MKL_INT head_size;

	Logger *loggers;
	pthread_barrier_t *barrier_attention_start;
	pthread_barrier_t *barrier_attention_end;

	/// Ops (set by caller)
	layer_dense *ql;
	layer_dense *kl;
	layer_dense *vl;
	matmul *score_attention;
	normalize *norm_attention;
	softmax *score_softmax;
	partial_softmax *partial_score_softmax;
	matmul *weighted_sum;

	/// buffers (set by caller)
	void *multi_b_query;
	void *multi_b_key;
	void *multi_b_value;
	void *multi_attention_score;
	void *multi_attention_softmax;
	void *multi_attention_result;
	void *multi_attention_result_col_tmp;
	void *m_attention_mask;
	void *col_split_m_attention_mask;
	void *partial_sum;

	/// barriers (set by caller)
	pthread_barrier_t *barrier_attention_q_gen;
	pthread_barrier_t *barrier_attention_k_gen;
	pthread_barrier_t *barrier_attention_score_cal;
	pthread_barrier_t *barrier_attention_score_norm;
	pthread_barrier_t *barrier_attention_mask_sub;
	pthread_barrier_t *barrier_attention_softmax;
	pthread_barrier_t *barrier_attention_v_gen;
	pthread_barrier_t *barrier_attention_weighted_sum;

	/// Input (set by caller)
	float *input;
	float ***col_split_input;    // only for column-based alg.
	int layer_idx;
	int batch_idx;
};

// Object for communication with ffw threads
struct Feedforward_State {
	EXEC_MODE exec_mode;
	int batch_tid;
	int head_tid;

	int num_batchs;
	int batch_th_num;
	int num_heads;  // FIXME: feedforward should be independently chunked, not based on # of heads
	int head_th_num;
	int num_layers;
//	int num_ffw_chunks;
	MKL_INT seq_length;
	MKL_INT hidden_size;
	MKL_INT head_size;
	MKL_INT feedforward_size;
	MKL_INT ffw_chunk_size;

	Logger *loggers;
	pthread_barrier_t *barrier_ffw_start;
	pthread_barrier_t *barrier_ffw_end;

	/// Ops (set by caller)
	layer_dense *attention_fc;
	layer_dense *prev_feedforward;
	gelu *gelu_feedforward;
	layer_dense *post_feedforward;

	/// buffers (set by caller)
	void *attention_result_split_tmp;
	void *feedforward_intermediate;
	void *feedforward_gelu;
	void *feedforward_result_split_tmp;

	/// barriers (set by caller)
	pthread_barrier_t *barrier_tfenc_attention_fc;
	pthread_barrier_t *barrier_tfenc_feedforward_pre;
	pthread_barrier_t *barrier_tfenc_feedforward_gelu;
	pthread_barrier_t *barrier_tfenc_feedforward_post;

	/// Input (set by caller)
	float ***head_split_context;    // [batch][heads][seq x head_size]

	float **attention_layernorm;
	int layer_idx;
	int batch_idx;
};

struct BERT_State {
	int vocab_size;
	int token_size;
	int num_batch;
	int num_heads;
	int seq_length;
	int hidden_size;
	int feedforwardsize;
	int num_layer;
	bool column_based;
	int chunk_size;
	int num_chunks;

	float **emb_weight;
	float ***weight;
	float **pooler_weight;
	int **input;
	float **mask;
	int **token_type;

	Attention_State **att_st_list;
	Feedforward_State **ffw_st_list;
	pthread_barrier_t *barrier_attention_start;
	pthread_barrier_t *barrier_attention_end;
	pthread_barrier_t *barrier_ffw_start;
	pthread_barrier_t *barrier_ffw_end;
};

#define BERT_NUM_LAYER         (24)
#define BERT_NUM_EMBED_WEIGHT   (5)
#define BERT_NUM_POOLER_WEIGHT  (2)
//#define BERT_NUM_WEIGHT   (16)

void init_bert_state(Params *params, BERT_State *bert_state);
void deinit_bert_state(Params *params, BERT_State *bert_state);
float *load_pooled_output(Params *params, BERT_State *bert_state);
#endif //MODEL_BERT_STATE_HPP
