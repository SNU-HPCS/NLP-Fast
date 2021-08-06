#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP
#include "tensor_func.hpp"
#include "log.hpp"
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

	bool is_training;
	int batch;
	MKL_INT hidden_size;
	MKL_INT seq_length;

	matmul *onehot_matmul;
	matmul *token_matmul;
	layer_norm *embedding_layernorm;


public:
	embedding(bool _is_training, float *_embedding_table, int _vocab_size, bool _use_onehot, float *_token_table, float *_position_table, bool _use_token_type, bool _use_position_embedding, float* gamma, float* beta, int _token_type_vocab_size, int _batch, MKL_INT _hidden_size, MKL_INT _seq_length);
	void embedding_lookup(int *input, int batch_idx, float *output, Logger *logger);
	void embedding_postprocessor(float *input, int *token_id, int id, float *output, Logger *logger);
	void embedding_deinit();
};

#endif
