#include <cstring>

#include "tensor_op_cpu.hpp"
#include "bert_op_cpu.hpp"

void create_attention_mask_from_input(BERT_State *bert_state) {
	const MKL_INT q_seq_length = bert_state->seq_length;
	const MKL_INT k_seq_length = bert_state->seq_length;
	const int num_batch = bert_state->num_batch;
	float **input_mask = bert_state->mask;
	float *one_mat = nullptr;

	/// attention_mask
	matmul mm = matmul(q_seq_length, k_seq_length, 1);
	for (int i = 0; i < num_batch; i++) {
		mm.forward(bert_state->ones_for_attention_mask[i], input_mask[i], false, bert_state->attention_mask[i]);
	}

	/// m_attention_mask
	one_mat = (float*)mkl_calloc(bert_state->seq_length, bert_state->seq_length * sizeof(float), 64);
	for (int i = 0; i < bert_state->seq_length * bert_state->seq_length; i++) { one_mat[i] = 1.0f; }
	for (int i = 0; i < num_batch; i++) {
		vsSub(bert_state->seq_length * bert_state->seq_length, one_mat, bert_state->attention_mask[i],
				bert_state->m_attention_mask[i]);
		cblas_sscal(bert_state->seq_length * bert_state->seq_length, 10000.0, bert_state->m_attention_mask[i], 1);
	}
	mkl_free(one_mat);
}

void apply_embedding(BERT_State *bert_state) {
	const int num_batch = bert_state->num_batch;
	const MKL_INT seq_length = bert_state->seq_length;
	const MKL_INT hidden_size = bert_state->hidden_size;
	embedding *embedding_layer = nullptr;
	float **embedding_output_lookup;

	/// Init values & embedding obj
	embedding_output_lookup=(float**)mkl_calloc(num_batch, sizeof(float*), 64);
	for(int i=0; i<num_batch; i++){
		embedding_output_lookup[i] = (float*)mkl_calloc(seq_length, hidden_size*sizeof(float), 64);
		memset(embedding_output_lookup[i], 0, sizeof(float) * seq_length * hidden_size);
	}

	embedding_layer = new embedding(bert_state->emb_weight[0], bert_state->vocab_size, true,
			bert_state->emb_weight[1], bert_state->emb_weight[2], true, true,
			bert_state->emb_weight[3], bert_state->emb_weight[4], bert_state->token_size, num_batch, hidden_size, seq_length);

	/// Do embedding
	for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
		embedding_layer->embedding_lookup(bert_state->input[batch_idx], batch_idx, embedding_output_lookup[batch_idx]);
		embedding_layer->embedding_postprocessor(embedding_output_lookup[batch_idx], bert_state->token_type[batch_idx],
												 batch_idx, bert_state->embedding_output[batch_idx]);
	}

	/// [DONE] Free objects
	for (int i = 0; i < num_batch; i++) {
		mkl_free(embedding_output_lookup[i]);
	}
	mkl_free(embedding_output_lookup);
	embedding_layer->embedding_deinit();
	delete(embedding_layer);
}

embedding::embedding(float *_embedding_table, int _vocab_size, bool _use_onehot,
					 float *_token_table, float *_position_table, bool _use_token_type, bool _use_position_embedding,
					 float* gamma, float* beta, int _token_type_vocab_size, int _batch, MKL_INT _hidden_size, MKL_INT _seq_length){
	embedding_table = _embedding_table;
	vocab_size = _vocab_size;
	use_onehot = _use_onehot;
	token_table = _token_table;
	position_table = _position_table;
	use_token_type = _use_token_type;
	use_position_embedding = _use_position_embedding;
	token_type_vocab_size = _token_type_vocab_size;
	batch = _batch;
	hidden_size = _hidden_size;
	seq_length = _seq_length;

	if(use_onehot){
		onehot_matmul = new matmul(seq_length, hidden_size, vocab_size);
	}
	///token size///
	if(use_token_type){
		token_matmul = new matmul(seq_length, hidden_size, token_type_vocab_size);
	}
	embedding_layernorm = new layer_norm(gamma, beta, batch, seq_length, hidden_size);

	if(use_position_embedding){
		position_embedding = (float*)mkl_calloc(seq_length, hidden_size*sizeof(float), 64);
	} else {
		position_embedding = nullptr;
	}
	onehot_id = (float **)mkl_calloc(batch, sizeof(float *), 64);
	onehot_token_id = (float **)mkl_calloc(batch, sizeof(float *), 64);
	token_embedding = (float **)mkl_calloc(batch, sizeof(float *), 64);
	postprocess_intermediate = (float **)mkl_calloc(batch, sizeof(float*), 64);
	before_layernorm = (float **)mkl_calloc(batch, sizeof(float*), 64);
	for(int i=0; i<batch; i++){
		onehot_id[i] = (float*)mkl_calloc(seq_length, vocab_size*sizeof(float), 64);
		onehot_token_id[i] = (float*)mkl_calloc(seq_length, token_type_vocab_size*sizeof(float), 64);
		postprocess_intermediate[i] = (float*)mkl_calloc(seq_length, hidden_size*sizeof(float), 64);
		token_embedding[i] = (float*)mkl_calloc(seq_length, hidden_size*sizeof(float), 64);
		before_layernorm[i] = (float*)mkl_calloc(seq_length, hidden_size*sizeof(float), 64);

		memset(onehot_id[i], 0, sizeof(float) * seq_length * vocab_size);
		memset(onehot_token_id[i], 0, sizeof(float) * seq_length * token_type_vocab_size);
		memset(token_embedding[i], 0, sizeof(float) * seq_length * hidden_size);
		memset(postprocess_intermediate[i], 0, sizeof(float) * seq_length * hidden_size);
		memset(before_layernorm[i], 0, sizeof(float) * seq_length * hidden_size);
	}
}

void embedding::embedding_lookup(int *input, int batch_idx, float *output) {
	if(use_onehot){
		one_hot(seq_length, vocab_size, input, onehot_id[batch_idx]);
		onehot_matmul->forward(onehot_id[batch_idx], embedding_table, false, output);
	}
	else{//gather
		//output = gather(input, embedding_table, seq_length, hidden_size);
		gather(input, embedding_table, seq_length, hidden_size, output);
	}
}

void embedding::embedding_postprocessor(float *input, int *token_id, int id, float *output) {
	if(use_token_type){
		if(token_id==nullptr)
			return;

		one_hot(seq_length, token_type_vocab_size, token_id, onehot_token_id[id]);
		token_matmul->forward(onehot_token_id[id], token_table, false, token_embedding[id]);
		add(token_embedding[id], input, seq_length*hidden_size, postprocess_intermediate[id]);
	}
	else{
		cblas_scopy(seq_length*hidden_size, postprocess_intermediate[id], 1, input, 1);
	}

	if(use_position_embedding){
		cblas_scopy((seq_length)*hidden_size, position_table, 1, position_embedding, 1);
		add(postprocess_intermediate[id], position_embedding, seq_length*hidden_size, before_layernorm[id]);
	}
	else{
		cblas_scopy(seq_length*hidden_size, before_layernorm[id], 1, postprocess_intermediate[id], 1);
	}
	embedding_layernorm->forward(before_layernorm[id], id, output);
}

void embedding::embedding_deinit() {

	for(int i=0; i<batch; i++){
		mkl_free(onehot_id[i]);
		mkl_free(onehot_token_id[i]);
		mkl_free(postprocess_intermediate[i]);
		mkl_free(token_embedding[i]);
		mkl_free(before_layernorm[i]);
	}
	mkl_free(onehot_id);
	mkl_free(onehot_token_id);
	mkl_free(token_embedding);
	mkl_free(postprocess_intermediate);
	mkl_free(before_layernorm);

	if (position_embedding) mkl_free(position_embedding);

	delete(onehot_matmul);
	delete(token_matmul);
	embedding_layernorm->layer_norm_deinit();
	delete(embedding_layernorm);
}
