#ifndef GPU_MODEL_BERT_STATE_HPP
#define GPU_MODEL_BERT_STATE_HPP

#include "utils.hpp"
#include "log.hpp"

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

struct BERT_State {
	int vocab_size;
	int token_size;
	int num_batch;
	int num_heads;
	int seq_length;
	int hidden_size;
	int feedforwardsize;
	int num_layer;

	float **emb_weight;
	float ***weight;
	float ***weight_attention_fc_splitted;      // [num_layer][chunk_idx][head_size x hidden_size]
	float **weight_attention_fc_bias_splitted; // [num_layer][seq_len x hidden_size]
	float ***weight_ffw_prev_splitted;          // [num_layer][chunk_idx][hidden_size x ffw_chunk_size]
	float ***weight_ffw_prev_bias_splitted;     // [num_layer][chunk_idx][seq_len x ffw_chunk_size]
	float ***weight_ffw_post_splitted;          // [num_layer][chunk_idx][ffw_chunk_size x hidden_size]
	float **weight_ffw_post_bias_splitted;     // [num_layer][seq_len x hidden_size]
	float **pooler_weight;
	int **input;
	float **mask;
	int **token_type;

	float **ones_for_attention_mask;
	float **attention_mask;
	float **m_attention_mask;
	float **embedding_output;
};

#define BERT_NUM_LAYER         (24)
#define BERT_NUM_EMBED_WEIGHT   (5)
#define BERT_NUM_POOLER_WEIGHT  (2)
//#define BERT_NUM_WEIGHT   (16)

void init_bert_state(Params *params, BERT_State *bert_state);
void deinit_bert_state(Params *params, BERT_State *bert_state);
float *load_pooled_output(Params *params, BERT_State *bert_state);
#endif //GPU_MODEL_BERT_STATE_HPP
