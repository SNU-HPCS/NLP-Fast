#ifndef MODEL_COLUMN_MODEL_HPP
#define MODEL_COLUMN_MODEL_HPP

#include <pthread.h>

#include "bert_state.hpp"
#include "tensor_func.hpp"
#include "embedding.hpp"
#include "log.hpp"

//////////////////////
/// self_attention_column
//////////////////////
class self_attention_column{
private:
	Params *params;
	BERT_State *bert_state;
	bool is_training;
	int num_heads;
	int num_th_head;
	int num_chunks;

	pthread_barrier_t *barrier_attention_q_gen;
	pthread_barrier_t *barrier_attention_k_gen;
	pthread_barrier_t *barrier_attention_score_cal;
	pthread_barrier_t *barrier_attention_score_norm;
	pthread_barrier_t *barrier_attention_mask_sub;
	pthread_barrier_t *barrier_attention_softmax;
	pthread_barrier_t *barrier_attention_v_gen;
	pthread_barrier_t *barrier_attention_weighted_sum;

	float *one_mat;
//	float **query_matrix;
//	float **key_matrix;
//	float **value_matrix;

	/// [head][weight (Hidden x Hidden/heads)]
	float **weight_qlw_splitted;
	float **weight_klw_splitted;
	float **weight_vlw_splitted;

	/// [head][weight (1 x Hidden/heads)]
	float **bias_qlw_splitted;
	float **bias_klw_splitted;
	float **bias_vlw_splitted;

	/// [batch][num_chunks][chunk_size x hidden]
	float ***col_split_input;
	/// [batch][head][seq x head_size]
	float ***multi_b_query;
	/// [batch][head][num_chunk][chunk_size x head_size]
	float ****multi_b_key;
	float ****multi_b_value;
	/// [batch][head][num_chunk][seq_length x chunk_size]
	float ****multi_attention_score;
	float ****multi_attention_softmax;
	/// [batch][head][seq_length x head_size]
	float ***multi_attention_result;
	/// [batch][head][num_chunk][seq_length x head_size]
	float ****multi_attention_result_col_tmp;
	/// [batch][seq_length x seq_length]
	float **m_attention_mask;
	/// [batch][num_chunks][seq_length x chunk_size]
	float ***col_split_m_attention_mask;
	/// [batch][head][seq_length x 1]
	float ***partial_sum;
	float **output;
	int batch;
	MKL_INT seq_length;
	MKL_INT hidden_size;
	MKL_INT head_size;
	MKL_INT chunk_size;

	layer_dense *ql;
	layer_dense *kl;
	layer_dense *vl;
	matmul *score_attention;
	normalize *norm_attention;
	partial_softmax *partial_score_softmax;
	matmul *weighted_sum;

	void init_weight_splitted(BERT_State *bert_state, int layer_idx);

public:
	self_attention_column(Params *_params, BERT_State *bert_state, int layer_idx, bool _is_training);
	void dump_values();
	float *forward(int batch_tid, float *input, float *attention_mask, int batch_idx, int layer_idx, Logger *loggers);
	void create_attention_mask(float* attention_mask, float* mask);
	void self_attention_deinit();
};

//////////////////////
/// transformer_encoder_column
//////////////////////
class transformer_encoder_column{
private:
	Params *params;
	bool is_training;
	int batch;
	int num_heads;
	MKL_INT seq_length;
	MKL_INT hidden_size;
	MKL_INT feedforward_size;

	pthread_barrier_t *barrier_tfenc_attention_layer;
	pthread_barrier_t *barrier_tfenc_attention_fc;
	pthread_barrier_t *barrier_tfenc_attention_residual;
	pthread_barrier_t *barrier_tfenc_attention_layernorm;
	pthread_barrier_t *barrier_tfenc_feedforward_pre;
	pthread_barrier_t *barrier_tfenc_feedforward_gelu;
	pthread_barrier_t *barrier_tfenc_feedforward_post;
	pthread_barrier_t *barrier_tfenc_feedforward_residual;
	pthread_barrier_t *barrier_tfenc_feedforward_layernorm;

	self_attention_column *attention_layer;
	layer_dense *attention_fc;
	layer_norm attention_norm;
	layer_dense *prev_feedforward;
	gelu gelu_feedforward;
	layer_dense *post_feedforward;
	layer_norm feedforward_norm;

	float **context;
	float **attention_result;
	float **attention_residual;
	float **attention_layernorm;
	float **feedforward_intermediate;
	float **feedforward_gelu;
	float **feedforward_result;
	float **feedforward_residual;
	float **feedforward_layernorm;

public:
	transformer_encoder_column(Params *_params, BERT_State *bert_state, int layer_idx, bool _is_training);
	void dump_values();
	float *forward(int batch_tid, float *a, float *attention_mask, int batch_idx, int layer_idx, Logger *loggers);
	void transformer_encoder_deinit();
};


//////////////////////
/// BERT
//////////////////////
class BERT{
private:
	Params *params;
	bool is_training;
	int batch;
	int batch_thread_num;
	int num_layers;
	int num_heads;
	int head_thread_num;
	MKL_INT seq_length;
	MKL_INT hidden_size;

	pthread_barrier_t *barrier_bert_embedding_lookup;
	pthread_barrier_t *barrier_bert_embedding_postprocessor;
	pthread_barrier_t *barrier_bert_transformer_layer;

	//embedding embedding_layer;
	embedding *embedding_layer;
	transformer_encoder_column **transformer_layer;
	layer_dense *pooler;

	BERT_State *bert_state;
	float ***layer_output;
	float **output;
	float **attention_mask;
	float **ones_for_attention_mask;
	float *first_token_tensor;
	float *pooler_output;
public:
	float **embedding_output_lookup;
	float **embedding_output;
	BERT(Params *_params, BERT_State *_bert_state, bool _is_training, bool use_onehot, bool use_token_type, bool use_position_embedding);
	void dump_values();
	float* forward(int ** input_ids, float **input_mask, int **token_type_ids, Logger *loggers);
	void create_attention_mask_from_input(float **attention_mask, MKL_INT q_seq_length, MKL_INT k_seq_length, int num_batch);
	void *forward_pth(int batch_tid, int id, int ** input, float ** mask, int ** token_type, float ** output, Logger *loggers);
	void BERT_deinit();
};
#endif //MODEL_COLUMN_MODEL_HPP
