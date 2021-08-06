#ifndef MODEL_BERT_OP_CPU_HPP
#define MODEL_BERT_OP_CPU_HPP
#include <mkl.h>
#include "tensor_op_cpu.hpp"
#include "bert_state.hpp"

void create_attention_mask_from_input(BERT_State *bert_state);
void apply_embedding(BERT_State *bert_state);

class embedding{
private:
//for embedding lookup
	float *embedding_table;
	int vocab_size;
	bool use_onehot;

	float **onehot_id;

//embedding post processing

	float *token_table;
	float *position_table;
	bool use_token_type;
	bool use_position_embedding;
	int token_type_vocab_size;

	float **onehot_token_id;
	float **token_embedding;
	float *position_embedding;
	float **postprocess_intermediate;
	float **before_layernorm;

	int batch;
	MKL_INT hidden_size;
	MKL_INT seq_length;

	matmul *onehot_matmul;
	matmul *token_matmul;
	layer_norm *embedding_layernorm;


public:
	embedding(float *_embedding_table, int _vocab_size, bool _use_onehot, float *_token_table, float *_position_table, bool _use_token_type, bool _use_position_embedding, float* gamma, float* beta, int _token_type_vocab_size, int _batch, MKL_INT _hidden_size, MKL_INT _seq_length);
	void embedding_lookup(int *input, int batch_idx, float *output);
	void embedding_postprocessor(float *input, int *token_id, int id, float *output);
	void embedding_deinit();
};

#endif //MODEL_BERT_OP_CPU_HPP
