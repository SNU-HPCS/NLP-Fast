#include <thread>
#include <mkl.h>
#include <cstring>
#include <cmath>
#include <sys/sysinfo.h>
#include <sched.h>

#include "utils.hpp"
#include "column_model.hpp"

void self_attention_column::init_weight_splitted(BERT_State *bert_state, int layer_idx) {
	weight_qlw_splitted = (float **)mkl_calloc(num_heads, sizeof(float*), 64);
	weight_klw_splitted = (float **)mkl_calloc(num_heads, sizeof(float*), 64);
	weight_vlw_splitted = (float **)mkl_calloc(num_heads, sizeof(float*), 64);
	bias_qlw_splitted= (float **)mkl_calloc(num_heads, sizeof(float*), 64);
	bias_klw_splitted = (float **)mkl_calloc(num_heads, sizeof(float*), 64);
	bias_vlw_splitted = (float **)mkl_calloc(num_heads, sizeof(float*), 64);
	for (int h = 0; h < num_heads; h++) {
		weight_qlw_splitted[h] = (float*)mkl_calloc(hidden_size, head_size * sizeof(float), 64);
		weight_klw_splitted[h] = (float*)mkl_calloc(hidden_size, head_size * sizeof(float), 64);
		weight_vlw_splitted[h] = (float*)mkl_calloc(hidden_size, head_size * sizeof(float), 64);
		bias_qlw_splitted[h] = (float*)mkl_calloc(1, head_size * sizeof(float), 64);
		bias_klw_splitted[h] = (float*)mkl_calloc(1, head_size * sizeof(float), 64);
		bias_vlw_splitted[h] = (float*)mkl_calloc(1, head_size * sizeof(float), 64);
	}

	for (int h = 0; h < num_heads; h++) {
		for (int k = 0; k < hidden_size; k++) {
			memcpy(&weight_qlw_splitted[h][k * head_size], &bert_state->weight[layer_idx][WEIGHT_QLW][k * hidden_size + h * head_size], head_size * sizeof(float));
			memcpy(&weight_klw_splitted[h][k * head_size], &bert_state->weight[layer_idx][WEIGHT_KLW][k * hidden_size + h * head_size], head_size * sizeof(float));
			memcpy(&weight_vlw_splitted[h][k * head_size], &bert_state->weight[layer_idx][WEIGHT_VLW][k * hidden_size + h * head_size], head_size * sizeof(float));
		}
		memcpy(bias_qlw_splitted[h], &bert_state->weight[layer_idx][WEIGHT_QLB][h * head_size], head_size * sizeof(float));
		memcpy(bias_klw_splitted[h], &bert_state->weight[layer_idx][WEIGHT_KLB][h * head_size], head_size * sizeof(float));
		memcpy(bias_vlw_splitted[h], &bert_state->weight[layer_idx][WEIGHT_VLB][h * head_size], head_size * sizeof(float));
	}
}

self_attention_column::self_attention_column(Params *_params, BERT_State *_bert_state, int layer_idx, bool _is_training) {
	params = _params;
	bert_state = _bert_state;
	batch = bert_state->num_batch;
	is_training = _is_training;
	num_heads = bert_state->num_heads;
	seq_length = bert_state->seq_length;
	hidden_size = bert_state->hidden_size;
	head_size = hidden_size / num_heads;
	num_th_head = params->num_th_head;
	num_chunks = bert_state->num_chunks;
	chunk_size = bert_state->chunk_size;
	assert(chunk_size != 0);
	assert(num_chunks != 0);

	barrier_attention_q_gen = new pthread_barrier_t;
	barrier_attention_k_gen = new pthread_barrier_t;
	barrier_attention_score_cal = new pthread_barrier_t;
	barrier_attention_score_norm = new pthread_barrier_t;
	barrier_attention_mask_sub = new pthread_barrier_t;
	barrier_attention_softmax = new pthread_barrier_t;
	barrier_attention_v_gen = new pthread_barrier_t;
	barrier_attention_weighted_sum = new pthread_barrier_t;
	pthread_barrier_init(barrier_attention_q_gen, nullptr, num_th_head * params->num_th_batch);
	pthread_barrier_init(barrier_attention_k_gen, nullptr, num_th_head * params->num_th_batch);
	pthread_barrier_init(barrier_attention_score_cal, nullptr, num_th_head * params->num_th_batch);
	pthread_barrier_init(barrier_attention_score_norm, nullptr, num_th_head * params->num_th_batch);
	pthread_barrier_init(barrier_attention_mask_sub, nullptr, num_th_head * params->num_th_batch);
	pthread_barrier_init(barrier_attention_softmax, nullptr, num_th_head * params->num_th_batch);
	pthread_barrier_init(barrier_attention_v_gen, nullptr, num_th_head * params->num_th_batch);
	pthread_barrier_init(barrier_attention_weighted_sum, nullptr, num_th_head * params->num_th_batch);

	/// Init QKV weight and bias
	init_weight_splitted(bert_state, layer_idx);

	/// Init ops
	ql = (layer_dense *)mkl_calloc(num_heads, sizeof(layer_dense), 64);
	kl = (layer_dense *)mkl_calloc(num_heads, sizeof(layer_dense), 64);
	vl = (layer_dense *)mkl_calloc(num_heads, sizeof(layer_dense), 64);
	score_attention = (matmul *)mkl_calloc(num_heads, sizeof(matmul), 64);
	norm_attention = (normalize *)mkl_calloc(num_heads, sizeof(normalize), 64);
	partial_score_softmax = (partial_softmax *)mkl_calloc(num_heads, sizeof(partial_softmax), 64);
	weighted_sum = (matmul *)mkl_calloc(num_heads, sizeof(matmul), 64);
	for (int h = 0; h < num_heads; h++) {
		ql[h] = layer_dense(weight_qlw_splitted[h], bias_qlw_splitted[h],
		                    is_training, seq_length, head_size, hidden_size);
		kl[h] = layer_dense(weight_klw_splitted[h], bias_klw_splitted[h],
		                    is_training, chunk_size, head_size, hidden_size);
		vl[h] = layer_dense(weight_vlw_splitted[h], bias_vlw_splitted[h],
		                    is_training, chunk_size, head_size, hidden_size);

		score_attention[h] = matmul(is_training, seq_length, chunk_size, head_size);
		norm_attention[h] = normalize();
		partial_score_softmax[h] = partial_softmax(is_training, seq_length, chunk_size);
		weighted_sum[h] = matmul(is_training, seq_length, head_size, chunk_size);
	}

	one_mat = (float *)mkl_calloc(seq_length, seq_length*sizeof(float), 64);

	col_split_input = (float ***)mkl_calloc(batch, sizeof(float*), 64);
	multi_b_query = (float ***)mkl_calloc(batch, sizeof(float **), 64);
	multi_b_key = (float ****)mkl_calloc(batch, sizeof(float ***), 64);
	multi_b_value = (float ****)mkl_calloc(batch, sizeof(float ***), 64);
	multi_attention_score = (float ****)mkl_calloc(batch, sizeof(float ***), 64);
	multi_attention_softmax = (float ****)mkl_calloc(batch, sizeof(float ***), 64);
	multi_attention_result = (float ***)mkl_calloc(batch, sizeof(float **), 64);
	multi_attention_result_col_tmp = (float ****)mkl_calloc(batch, sizeof(float ***), 64);
	m_attention_mask=(float **)mkl_calloc(batch, sizeof(float*), 64);
	col_split_m_attention_mask = (float ***)mkl_calloc(batch, sizeof(float**), 64);
	partial_sum = (float***)mkl_calloc(batch, sizeof(float **), 64);
	output = (float**)mkl_calloc(batch, sizeof(float*), 64);
	ones(seq_length, seq_length, one_mat);
	for(int i=0; i<batch; i++){
		col_split_input[i] = (float **)mkl_calloc(num_chunks, sizeof(float*), 64);
		multi_b_query[i] = (float **)mkl_calloc(num_heads, sizeof(float*), 64);
		multi_b_key[i] = (float ***)mkl_calloc(num_heads, sizeof(float**), 64);
		multi_b_value[i] = (float ***)mkl_calloc(num_heads, sizeof(float**), 64);
		multi_attention_score[i] = (float ***) mkl_calloc(num_heads, sizeof(float**), 64);
		multi_attention_softmax[i] = (float ***)mkl_calloc(num_heads, sizeof(float**), 64);
		multi_attention_result[i] = (float **) mkl_calloc(num_heads, sizeof(float*), 64);
		multi_attention_result_col_tmp[i] = (float ***) mkl_calloc(num_heads, sizeof(float**), 64);
		m_attention_mask[i] = (float *)mkl_calloc(seq_length, seq_length*sizeof(float), 64);
		memset(m_attention_mask[i], 0, sizeof(float) * seq_length * seq_length);
		col_split_m_attention_mask[i] = (float **)mkl_calloc(num_chunks, sizeof(float*), 64);
		for (int c = 0; c < num_chunks; c++) {
			col_split_input[i][c] = (float *)mkl_calloc(chunk_size, hidden_size * sizeof(float), 64);
			col_split_m_attention_mask[i][c] = (float *)mkl_calloc(seq_length, chunk_size * sizeof(float), 64);
			memset(col_split_input[i][c], 0, sizeof(float) * seq_length * chunk_size);
			memset(col_split_m_attention_mask[i][c], 0, sizeof(float) * seq_length * chunk_size);
		}
		partial_sum[i] = (float **)mkl_calloc(num_heads, sizeof(float*), 64);
		output[i] = (float *)mkl_calloc(seq_length, hidden_size*sizeof(float), 64);
		memset(output[i],           0, sizeof(float) * seq_length * hidden_size);

		for(int j=0; j<num_heads; j++){
			multi_b_query[i][j] = (float*)mkl_calloc(seq_length, head_size*sizeof(float), 64);
			partial_sum[i][j] = (float*)mkl_calloc(seq_length, 1 * sizeof(float), 64);
			multi_attention_result[i][j] = (float*)mkl_calloc(seq_length, head_size * sizeof(float), 64);
			memset(multi_b_query[i][j],           0, sizeof(float) * seq_length * head_size);
			memset(partial_sum[i][j],             0, sizeof(float) * seq_length * 1);
			memset(multi_attention_result[i][j],  0, sizeof(float) * seq_length * head_size);

			multi_b_key[i][j] = (float**)mkl_calloc(num_chunks, sizeof(float*), 64);
			multi_b_value[i][j] = (float**)mkl_calloc(num_chunks, sizeof(float*), 64);
			multi_attention_score[i][j] = (float**)mkl_calloc(num_chunks, sizeof(float*), 64);
			multi_attention_softmax[i][j] = (float**)mkl_calloc(num_chunks, sizeof(float*), 64);
			multi_attention_result_col_tmp[i][j] = (float**)mkl_calloc(num_chunks, sizeof(float*), 64);
			for (int c = 0; c < num_chunks; c++) {
				multi_b_key[i][j][c] = (float *) mkl_calloc(chunk_size, head_size * sizeof(float), 64);
				multi_b_value[i][j][c] = (float *) mkl_calloc(chunk_size, head_size * sizeof(float), 64);
				multi_attention_score[i][j][c] = (float *) mkl_calloc(seq_length, chunk_size * sizeof(float), 64);
				multi_attention_softmax[i][j][c] = (float *) mkl_calloc(seq_length, chunk_size * sizeof(float), 64);
				multi_attention_result_col_tmp[i][j][c] = (float *) mkl_calloc(seq_length, head_size * sizeof(float), 64);
				memset(multi_b_key[i][j][c], 0, sizeof(float) * chunk_size * head_size);
				memset(multi_b_value[i][j][c], 0, sizeof(float) * chunk_size * head_size);
				memset(multi_attention_softmax[i][j][c], 0, sizeof(float) * seq_length * chunk_size);
				memset(multi_attention_score[i][j][c], 0, sizeof(float) * seq_length * chunk_size);
				memset(multi_attention_result_col_tmp[i][j][c], 0, sizeof(float) * seq_length * head_size);
			}
		}
	}
}

void self_attention_column::create_attention_mask(float *attention_mask, float *mask) {
	vsSub(seq_length * seq_length, one_mat, attention_mask, mask);
	cblas_sscal(seq_length * seq_length, 10000.0, mask, 1);
}

void self_attention_column::dump_values() {
	printf("=== self_attention_baseline ===\n");
	for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
		for (int head_idx = 0; head_idx < num_heads; head_idx++) {
			printf("multi_b_query[%d][%d] => ", batch_idx, head_idx);
			for (int i = 0; i < 16; i++) { printf("%f ", multi_b_query[batch_idx][head_idx][i]); } printf("\n");
			printf("multi_b_key[%d][%d][0] => ", batch_idx, head_idx);
			for (int i = 0; i < 16; i++) { printf("%f ", multi_b_key[batch_idx][head_idx][0][i]); } printf("\n");
			printf("multi_b_value[%d][%d][0] => ", batch_idx, head_idx);
			for (int i = 0; i < 16; i++) { printf("%f ", multi_b_value[batch_idx][head_idx][0][i]); } printf("\n");
			printf("multi_attention_score[%d][%d][0] => ", batch_idx, head_idx);
			for (int i = 0; i < 16; i++) { printf("%f ", multi_attention_score[batch_idx][head_idx][0][i]); } printf("\n");
			printf("multi_attention_softmax[%d][%d][0] => ", batch_idx, head_idx);
			for (int i = 0; i < 16; i++) { printf("%f ", multi_attention_softmax[batch_idx][head_idx][0][i]); } printf("\n");
			printf("multi_attention_result[%d][%d] => ", batch_idx, head_idx);
			for (int i = 0; i < 16; i++) { printf("%f ", multi_attention_result[batch_idx][head_idx][i]); } printf("\n");
		}
	}
}

float *self_attention_column::forward(int batch_tid, float *input, float *attention_mask, int batch_idx, int layer_idx, Logger *loggers) {
	Logger &logger = loggers[Logger::get_logger_idx(batch_idx, -1)];
	std::thread **th_list = nullptr;

	flush_cache(m_attention_mask[batch_idx], seq_length * seq_length * sizeof(float));
	prefetch_data(m_attention_mask[batch_idx], seq_length * seq_length * sizeof(float));
	logger.infer_logging_begin(ATTENTION_LAYER_MASK_CREATION, layer_idx);
	create_attention_mask(attention_mask, m_attention_mask[batch_idx]);
	logger.infer_logging_end(ATTENTION_LAYER_MASK_CREATION, layer_idx);

	/// column-specific data split
	for (int seq_idx = 0; seq_idx < seq_length; seq_idx++) {
		for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
			memcpy(col_split_m_attention_mask[batch_idx][chunk_idx] + chunk_size * seq_idx,
					m_attention_mask[batch_idx] + seq_length * seq_idx + chunk_size * chunk_idx,
					chunk_size * sizeof(float));
		}
	}

	/// Setting parameters for attention_worker threads
	for (int head_tid = 0; head_tid < num_th_head; head_tid++) {
		Attention_State *att_st = bert_state->att_st_list[batch_tid * num_th_head + head_tid];
		att_st->ql = ql;
		att_st->kl = kl;
		att_st->vl = vl;
		att_st->score_attention = score_attention;
		att_st->norm_attention = norm_attention;
		att_st->partial_score_softmax = partial_score_softmax;
		att_st->weighted_sum = weighted_sum;

		att_st->multi_b_query = multi_b_query;
		att_st->multi_b_key = multi_b_key;
		att_st->multi_b_value = multi_b_value;
		att_st->multi_attention_score = multi_attention_score;
		att_st->multi_attention_softmax = multi_attention_softmax;
		att_st->multi_attention_result = multi_attention_result;
		att_st->multi_attention_result_col_tmp = multi_attention_result_col_tmp;
		att_st->col_split_m_attention_mask = col_split_m_attention_mask;
		att_st->partial_sum = partial_sum;

		att_st->barrier_attention_q_gen = barrier_attention_q_gen;
		att_st->barrier_attention_k_gen = barrier_attention_k_gen;
		att_st->barrier_attention_score_cal = barrier_attention_score_cal;
		att_st->barrier_attention_score_norm = barrier_attention_score_norm;
		att_st->barrier_attention_mask_sub = barrier_attention_mask_sub;
		att_st->barrier_attention_softmax = barrier_attention_softmax;
		att_st->barrier_attention_v_gen = barrier_attention_v_gen;
		att_st->barrier_attention_weighted_sum = barrier_attention_weighted_sum;

		att_st->input = input;
		att_st->col_split_input = col_split_input;
		att_st->layer_idx = layer_idx;
		att_st->batch_idx = batch_idx;
	}

	/// Send a starting signal to workers
	pthread_barrier_wait(bert_state->barrier_attention_start);

	/// Waiting for attention workers
	pthread_barrier_wait(bert_state->barrier_attention_end);

	/// Merge results into a single output
	for (int i = 0; i < num_heads; i++) {
		flush_cache(multi_attention_result[batch_idx][i], seq_length * head_size * sizeof(float));
		prefetch_data(multi_attention_result[batch_idx][i], seq_length * head_size * sizeof(float));
	}
	logger.infer_logging_begin(ATTENTION_LAYER_HEAD_MERGE, layer_idx);
	merge(multi_attention_result[batch_idx], seq_length, hidden_size, num_heads, output[batch_idx]);
	logger.infer_logging_end(ATTENTION_LAYER_HEAD_MERGE, layer_idx);

	return output[batch_idx];
}

static void *attention_threads_column(void *_state) {
	Attention_State *att_st = (Attention_State*)_state;
	const EXEC_MODE exec_mode = att_st->exec_mode;
	const int batch_tid = att_st->batch_tid;
	const int head_tid = att_st->head_tid;
	const int num_batchs = att_st->num_batchs;
	const int batch_th_num = att_st->batch_th_num;
	const int num_batch_per_thread = num_batchs / batch_th_num;
	const int num_heads = att_st->num_heads;
	const int head_th_num = att_st->head_th_num;
	const int num_head_per_thread = num_heads / head_th_num;
	const int num_layers = att_st->num_layers;
	const bool column_based = att_st->column_based;
	const int num_chunks = att_st->num_chunks;
	const MKL_INT chunk_size = att_st->chunk_size;
	const MKL_INT seq_length = att_st->seq_length;
	const MKL_INT hidden_size = att_st->hidden_size;
	const MKL_INT head_size = att_st->head_size;
	const int head_start_idx = head_tid * num_head_per_thread;

	Logger *loggers = att_st->loggers;

	layer_dense *ql = nullptr;
	layer_dense *kl = nullptr;
	layer_dense *vl = nullptr;
	matmul *score_attention = nullptr;
	normalize *norm_attention = nullptr;
	partial_softmax *partial_score_softmax = nullptr;
	matmul *weighted_sum = nullptr;

	float ***multi_b_query = nullptr;
	float ****multi_b_key = nullptr;
	float ****multi_b_value = nullptr;
	float ****multi_attention_score = nullptr;
	float ****multi_attention_softmax = nullptr;
	float ***multi_attention_result = nullptr;
	float ****multi_attention_result_col_tmp = nullptr;
	float ***col_split_m_attention_mask = nullptr;  // [batch][chunk][seq * chk]
	float ***partial_sum = nullptr;

	pthread_barrier_t *barrier_attention_q_gen = nullptr;
	pthread_barrier_t *barrier_attention_k_gen = nullptr;
	pthread_barrier_t *barrier_attention_score_cal = nullptr;
	pthread_barrier_t *barrier_attention_score_norm = nullptr;
	pthread_barrier_t *barrier_attention_mask_sub = nullptr;
	pthread_barrier_t *barrier_attention_softmax = nullptr;
	pthread_barrier_t *barrier_attention_v_gen = nullptr;
	pthread_barrier_t *barrier_attention_weighted_sum = nullptr;

	float *input = nullptr;
	float ***col_split_input = nullptr;
	int layer_idx = -1;
	int batch_idx = -1;

	int cur_idx = 0;

	mkl_set_num_threads_local(1);
	/// Cache init
	Logger::init_csmon(batch_tid, head_tid, false);	// Only track itself
	Logger::csmon_start(batch_tid, head_tid);

	/////////////////////
	/// Main routine
	/////////////////////
	while (true) {
		cur_idx++;

		/// Start attention calculation (waiting for an input)
		pthread_barrier_wait(att_st->barrier_attention_start);
		ql = att_st->ql;
		kl = att_st->kl;
		vl = att_st->vl;
		score_attention = att_st->score_attention;
		norm_attention = att_st->norm_attention;
		partial_score_softmax = att_st->partial_score_softmax;
		weighted_sum = att_st->weighted_sum;

		multi_b_query = (float***)att_st->multi_b_query;
		multi_b_key = (float****)att_st->multi_b_key;
		multi_b_value = (float****)att_st->multi_b_value;
		multi_attention_score = (float****)att_st->multi_attention_score;
		multi_attention_softmax = (float****)att_st->multi_attention_softmax;
		multi_attention_result = (float***)att_st->multi_attention_result;
		multi_attention_result_col_tmp = (float****)att_st->multi_attention_result_col_tmp;
		col_split_m_attention_mask = (float***)att_st->col_split_m_attention_mask;
		partial_sum = (float***)att_st->partial_sum;

		barrier_attention_q_gen = att_st->barrier_attention_q_gen;
		barrier_attention_k_gen = att_st->barrier_attention_k_gen;
		barrier_attention_score_cal = att_st->barrier_attention_score_cal;
		barrier_attention_score_norm = att_st->barrier_attention_score_norm;
		barrier_attention_mask_sub = att_st->barrier_attention_mask_sub;
		barrier_attention_softmax = att_st->barrier_attention_softmax;
		barrier_attention_v_gen = att_st->barrier_attention_v_gen;
		barrier_attention_weighted_sum = att_st->barrier_attention_weighted_sum;

		input = att_st->input;
		col_split_input = att_st->col_split_input;
		layer_idx = att_st->layer_idx;
		batch_idx = att_st->batch_idx;

		/// column-specific data split
		for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
			memcpy(col_split_input[batch_idx][chunk_idx], input + chunk_size * hidden_size * chunk_idx, chunk_size * hidden_size * sizeof(float));
		}

		for (int head_idx = head_start_idx;  head_idx < head_start_idx + num_head_per_thread; head_idx++) {
			Logger &logger = loggers[Logger::get_logger_idx(batch_idx, head_idx)];
			logger.set_tids(batch_tid, head_tid);
			logger.set_cs_counter(Logger::get_csmon(batch_tid, head_tid)->get_counter());


			/////////////
			/// Q_GEN
			/////////////
			flush_cache(ql[head_idx].weight, hidden_size * head_size * sizeof(float));
			flush_cache(ql[head_idx].b_add.bias, 1 * head_size * sizeof(float));
			flush_cache(multi_b_query[batch_idx][head_idx], seq_length * head_size * sizeof(float));
			prefetch_data(ql[head_idx].weight, hidden_size * head_size * sizeof(float));
			prefetch_data(ql[head_idx].b_add.bias, 1 * head_size * sizeof(float));
			prefetch_data(multi_b_query[batch_idx][head_idx], seq_length * head_size * sizeof(float));
			pthread_barrier_wait(barrier_attention_q_gen);
			logger.infer_logging_begin(ATTENTION_LAYER_Q_GEN, layer_idx);
			ql[head_idx].forward(input, CblasNoTrans, multi_b_query[batch_idx][head_idx]);
			logger.infer_logging_end(ATTENTION_LAYER_Q_GEN, layer_idx);
			if (exec_mode == EXEC_MODE_PERF_TEST) {
				for (int i = 0; i < seq_length * head_size; i++) {
					multi_b_query[batch_idx][head_idx][i] = fmodf(multi_b_query[batch_idx][head_idx][i], VALUE_TUNING_QKV_THRESHOLD);
				}
			}

			memset(partial_sum[batch_idx][head_idx], 0, seq_length * sizeof(float));
			for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
				/////////////
				/// K_GEN
				/////////////
				flush_cache(col_split_input[batch_idx][chunk_idx], chunk_size * hidden_size * sizeof(float));
				flush_cache(kl[head_idx].weight, hidden_size * head_size * sizeof(float));
				flush_cache(kl[head_idx].b_add.bias, 1 * head_size * sizeof(float));
				flush_cache(multi_b_key[batch_idx][head_idx][chunk_idx], chunk_size * head_size * sizeof(float));
				prefetch_data(col_split_input[batch_idx][chunk_idx], chunk_size * hidden_size * sizeof(float));
				prefetch_data(kl[head_idx].weight, hidden_size * head_size * sizeof(float));
				prefetch_data(kl[head_idx].b_add.bias, 1 * head_size * sizeof(float));
				prefetch_data(multi_b_key[batch_idx][head_idx][chunk_idx], chunk_size * head_size * sizeof(float));
				pthread_barrier_wait(barrier_attention_k_gen);
				logger.infer_logging_begin(ATTENTION_LAYER_K_GEN, layer_idx);
				kl[head_idx].forward(col_split_input[batch_idx][chunk_idx], CblasNoTrans, multi_b_key[batch_idx][head_idx][chunk_idx]);
				logger.infer_logging_end(ATTENTION_LAYER_K_GEN, layer_idx);
				if (exec_mode == EXEC_MODE_PERF_TEST) {
					for (int i = 0; i < chunk_size * head_size; i++) {
						multi_b_key[batch_idx][head_idx][chunk_idx][i] = fmodf(multi_b_key[batch_idx][head_idx][chunk_idx][i], VALUE_TUNING_QKV_THRESHOLD);
					}
				}


				/////////////
				/// Score_calculation
				/////////////
				flush_cache(multi_b_query[batch_idx][head_idx], seq_length * head_size * sizeof(float));
				flush_cache(multi_b_key[batch_idx][head_idx][chunk_idx], chunk_size * head_size * sizeof(float));
				flush_cache(multi_attention_score[batch_idx][head_idx][chunk_idx], seq_length * chunk_size * sizeof(float));
				prefetch_data(multi_b_query[batch_idx][head_idx], seq_length * head_size * sizeof(float));
				prefetch_data(multi_b_key[batch_idx][head_idx][chunk_idx], chunk_size * head_size * sizeof(float));
				prefetch_data(multi_attention_score[batch_idx][head_idx][chunk_idx], seq_length * chunk_size * sizeof(float));
				pthread_barrier_wait(barrier_attention_score_cal);
				logger.infer_logging_begin(ATTENTION_LAYER_SCORE_CAL, layer_idx);
				score_attention[head_idx].forward(multi_b_query[batch_idx][head_idx], multi_b_key[batch_idx][head_idx][chunk_idx],
												  true, multi_attention_score[batch_idx][head_idx][chunk_idx]);
				logger.infer_logging_end(ATTENTION_LAYER_SCORE_CAL, layer_idx);
				if (exec_mode == EXEC_MODE_PERF_TEST) {
					for (int i = 0; i < seq_length * chunk_size; i++) {
						multi_attention_score[batch_idx][head_idx][chunk_idx][i] = fmodf(multi_attention_score[batch_idx][head_idx][chunk_idx][i], VALUE_TUNING_SCORE_THRESHOLD);
					}
				}


				/////////////
				/// socre_norm
				/////////////
				flush_cache(multi_attention_score[batch_idx][head_idx][chunk_idx], seq_length * chunk_size * sizeof(float));
				prefetch_data(multi_attention_score[batch_idx][head_idx][chunk_idx], seq_length * chunk_size * sizeof(float));
				pthread_barrier_wait(barrier_attention_score_norm);
				logger.infer_logging_begin(ATTENTION_LAYER_SCORE_NORM, layer_idx);
				norm_attention[head_idx].forward(multi_attention_score[batch_idx][head_idx][chunk_idx], seq_length * chunk_size,
												 1.0f / (float) head_size, true);
				logger.infer_logging_end(ATTENTION_LAYER_SCORE_NORM, layer_idx);


				/////////////
				/// ATTENTION_LAYER_MASK_SUB
				/////////////
				flush_cache(multi_attention_score[batch_idx][head_idx][chunk_idx], seq_length * chunk_size * sizeof(float));
				prefetch_data(multi_attention_score[batch_idx][head_idx][chunk_idx], seq_length * chunk_size * sizeof(float));
				pthread_barrier_wait(barrier_attention_mask_sub);
				logger.infer_logging_begin(ATTENTION_LAYER_MASK_SUB, layer_idx);
				if (exec_mode == EXEC_MODE_PERF_TEST) {
					// We don't apply masking on performance test (because we want to avoid high overhead of exponentiation on -9999 values)
				} else {
					vsSub(seq_length * chunk_size, multi_attention_score[batch_idx][head_idx][chunk_idx],
					      col_split_m_attention_mask[batch_idx][chunk_idx],
					      multi_attention_score[batch_idx][head_idx][chunk_idx]);
				}
				logger.infer_logging_end(ATTENTION_LAYER_MASK_SUB, layer_idx);


				/////////////
				/// Softmax
				/////////////
				flush_cache(multi_attention_score[batch_idx][head_idx][chunk_idx], seq_length * chunk_size * sizeof(float));
				flush_cache(multi_attention_softmax[batch_idx][head_idx][chunk_idx], seq_length * chunk_size * sizeof(float));
				flush_cache(partial_sum[batch_idx][head_idx], 1 * seq_length * sizeof(float));
				prefetch_data(multi_attention_score[batch_idx][head_idx][chunk_idx], seq_length * chunk_size * sizeof(float));
				prefetch_data(multi_attention_softmax[batch_idx][head_idx][chunk_idx], seq_length * chunk_size * sizeof(float));
				prefetch_data(partial_sum[batch_idx][head_idx], 1 * seq_length * sizeof(float));
				pthread_barrier_wait(barrier_attention_softmax);
				logger.infer_logging_begin(ATTENTION_LAYER_SOFTMAX, layer_idx);
				partial_score_softmax[head_idx].forward(multi_attention_score[batch_idx][head_idx][chunk_idx],
						multi_attention_softmax[batch_idx][head_idx][chunk_idx], partial_sum[batch_idx][head_idx]);
				logger.infer_logging_end(ATTENTION_LAYER_SOFTMAX, layer_idx);


				/////////////
				/// V_GEN
				/////////////
				flush_cache(col_split_input[batch_idx][chunk_idx], chunk_size * hidden_size * sizeof(float));
				flush_cache(vl[head_idx].weight, hidden_size * head_size * sizeof(float));
				flush_cache(vl[head_idx].b_add.bias, 1 * head_size * sizeof(float));
				flush_cache(multi_b_value[batch_idx][head_idx][chunk_idx], chunk_size * head_size * sizeof(float));
				prefetch_data(col_split_input[batch_idx][chunk_idx], chunk_size * hidden_size * sizeof(float));
				prefetch_data(vl[head_idx].weight, hidden_size * head_size * sizeof(float));
				prefetch_data(vl[head_idx].b_add.bias, 1 * head_size * sizeof(float));
				prefetch_data(multi_b_value[batch_idx][head_idx][chunk_idx], chunk_size * head_size * sizeof(float));
				pthread_barrier_wait(barrier_attention_v_gen);
				logger.infer_logging_begin(ATTENTION_LAYER_V_GEN, layer_idx);
				vl[head_idx].forward(col_split_input[batch_idx][chunk_idx], CblasNoTrans, multi_b_value[batch_idx][head_idx][chunk_idx]);
				logger.infer_logging_end(ATTENTION_LAYER_V_GEN, layer_idx);
				if (exec_mode == EXEC_MODE_PERF_TEST) {
					for (int i = 0; i < chunk_size * head_size; i++) {
						multi_b_value[batch_idx][head_idx][chunk_idx][i] = fmodf(multi_b_value[batch_idx][head_idx][chunk_idx][i], VALUE_TUNING_QKV_THRESHOLD);
					}
				}


				/////////////
				/// Weighted_Sum
				/////////////
				flush_cache(multi_attention_softmax[batch_idx][head_idx][chunk_idx], seq_length * chunk_size * sizeof(float));
				flush_cache(multi_b_value[batch_idx][head_idx][chunk_idx], chunk_size * head_size * sizeof(float));
				flush_cache(multi_attention_result_col_tmp[batch_idx][head_idx][chunk_idx], seq_length * head_size * sizeof(float));
				prefetch_data(multi_attention_softmax[batch_idx][head_idx][chunk_idx], seq_length * chunk_size * sizeof(float));
				prefetch_data(multi_b_value[batch_idx][head_idx][chunk_idx], chunk_size * head_size * sizeof(float));
				prefetch_data(multi_attention_result_col_tmp[batch_idx][head_idx][chunk_idx], seq_length * head_size * sizeof(float));
				pthread_barrier_wait(barrier_attention_weighted_sum);
				logger.infer_logging_begin(ATTENTION_LAYER_WEIGHTED_SUM, layer_idx);
				weighted_sum[head_idx].forward(multi_attention_softmax[batch_idx][head_idx][chunk_idx], multi_b_value[batch_idx][head_idx][chunk_idx],
											   false, multi_attention_result_col_tmp[batch_idx][head_idx][chunk_idx]);
				logger.infer_logging_end(ATTENTION_LAYER_WEIGHTED_SUM, layer_idx);
			}

			/////////////
			/// Lazy Softmax (FIXME: does it require latency & cachestat measurement?)
			/////////////
			/// Aggregate all results into chunk_idx: 0
			for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
				vsAdd(seq_length * head_size, multi_attention_result[batch_idx][head_idx], multi_attention_result_col_tmp[batch_idx][head_idx][chunk_idx],
				      multi_attention_result[batch_idx][head_idx]);
			}
			/// Divide all entries
			for (int seq_idx = 0; seq_idx < seq_length; seq_idx++) {
				for (int head_size_idx = 0; head_size_idx < head_size; head_size_idx++) {
					multi_attention_result[batch_idx][head_idx][seq_idx * head_size + head_size_idx] /= partial_sum[batch_idx][head_idx][seq_idx];
				}
			}


			/// Cache deinit
			logger.free_cs_counter();
		}

		/// Send a signal that the calculation is finished
		pthread_barrier_wait(att_st->barrier_attention_end);

		/// end condition
		if (cur_idx >= (num_batch_per_thread * num_layers)) {
			break;
		}
	}

	Logger::csmon_stop(batch_tid, head_tid);
	Logger::deinit_csmon(batch_tid, head_tid);

	return nullptr;
}

void self_attention_column::self_attention_deinit() {
	delete(barrier_attention_q_gen);
	delete(barrier_attention_k_gen);
	delete(barrier_attention_score_cal);
	delete(barrier_attention_score_norm);
	delete(barrier_attention_mask_sub);
	delete(barrier_attention_softmax);
	delete(barrier_attention_v_gen);
	delete(barrier_attention_weighted_sum);

	mkl_free(ql);
	mkl_free(kl);
	mkl_free(vl);
	mkl_free(score_attention);
	mkl_free(norm_attention);
	for (int h = 0; h < num_heads; h++) {
		partial_score_softmax[h].partial_softmax_deinit();
	}
	mkl_free(partial_score_softmax);
	mkl_free(weighted_sum);

	for (int h = 0; h < num_heads; h++) {
		mkl_free(weight_qlw_splitted[h]);
		mkl_free(weight_klw_splitted[h]);
		mkl_free(weight_vlw_splitted[h]);
		mkl_free(bias_qlw_splitted[h]);
		mkl_free(bias_klw_splitted[h]);
		mkl_free(bias_vlw_splitted[h]);
	}
	mkl_free(weight_qlw_splitted);
	mkl_free(weight_klw_splitted);
	mkl_free(weight_vlw_splitted);
	mkl_free(bias_qlw_splitted);
	mkl_free(bias_klw_splitted);
	mkl_free(bias_vlw_splitted);

	for(int i=0; i<batch; i++){
		for(int j=0; j<num_heads; j++){
			mkl_free(multi_b_query[i][j]);
			for (int c = 0; c < num_chunks; c++) {
				mkl_free(multi_b_key[i][j][c]);
				mkl_free(multi_b_value[i][j][c]);
				mkl_free(multi_attention_score[i][j][c]);
				mkl_free(multi_attention_softmax[i][j][c]);
				mkl_free(multi_attention_result_col_tmp[i][j][c]);
			}
			mkl_free(multi_b_key[i][j]);
			mkl_free(multi_b_value[i][j]);
			mkl_free(multi_attention_score[i][j]);
			mkl_free(multi_attention_softmax[i][j]);
			mkl_free(multi_attention_result[i][j]);
			mkl_free(multi_attention_result_col_tmp[i][j]);
			mkl_free(partial_sum[i][j]);
		}

		for (int c = 0; c <num_chunks; c++) {
			mkl_free(col_split_input[i][c]);
			mkl_free(col_split_m_attention_mask[i][c]);
		}

		mkl_free(col_split_input[i]);
		mkl_free(multi_b_query[i]);
		mkl_free(multi_b_key[i]);
		mkl_free(multi_b_value[i]);
		mkl_free(multi_attention_score[i]);
		mkl_free(multi_attention_softmax[i]);
		mkl_free(multi_attention_result[i]);
		mkl_free(multi_attention_result_col_tmp[i]);
		mkl_free(m_attention_mask[i]);
		mkl_free(col_split_m_attention_mask[i]);
		mkl_free(partial_sum[i]);
		mkl_free(output[i]);
	}
	mkl_free(one_mat); one_mat = nullptr;

	mkl_free(col_split_input); col_split_input = nullptr;
	mkl_free(multi_b_query); multi_b_query = nullptr;
	mkl_free(multi_b_key); multi_b_key = nullptr;
	mkl_free(multi_b_value); multi_b_value = nullptr;
	mkl_free(multi_attention_score); multi_attention_score = nullptr;
	mkl_free(multi_attention_softmax); multi_attention_softmax = nullptr;
	mkl_free(multi_attention_result); multi_attention_result = nullptr;
	mkl_free(multi_attention_result_col_tmp); multi_attention_result_col_tmp = nullptr;
	mkl_free(m_attention_mask); m_attention_mask = nullptr;
	mkl_free(col_split_m_attention_mask); col_split_m_attention_mask = nullptr;
	mkl_free(partial_sum); partial_sum = nullptr;
	mkl_free(output); output = nullptr;
}

transformer_encoder_column::transformer_encoder_column(Params *_params, BERT_State *bert_state, int layer_idx, bool _is_training) {
	params = _params;
	is_training = _is_training;
	batch = bert_state->num_batch;
	num_heads = bert_state->num_heads;
	seq_length = bert_state->seq_length;
	hidden_size = bert_state->hidden_size;
	feedforward_size = bert_state->feedforwardsize;

	barrier_tfenc_attention_layer = new pthread_barrier_t;
	barrier_tfenc_attention_fc = new pthread_barrier_t;
	barrier_tfenc_attention_residual = new pthread_barrier_t;
	barrier_tfenc_attention_layernorm = new pthread_barrier_t;
	barrier_tfenc_feedforward_pre = new pthread_barrier_t;
	barrier_tfenc_feedforward_gelu = new pthread_barrier_t;
	barrier_tfenc_feedforward_post = new pthread_barrier_t;
	barrier_tfenc_feedforward_residual = new pthread_barrier_t;
	barrier_tfenc_feedforward_layernorm = new pthread_barrier_t;
	pthread_barrier_init(barrier_tfenc_attention_layer, nullptr, params->num_th_batch);
	pthread_barrier_init(barrier_tfenc_attention_fc, nullptr, params->num_th_batch);
	pthread_barrier_init(barrier_tfenc_attention_residual, nullptr, params->num_th_batch);
	pthread_barrier_init(barrier_tfenc_attention_layernorm, nullptr, params->num_th_batch);
	pthread_barrier_init(barrier_tfenc_feedforward_pre, nullptr, params->num_th_batch);
	pthread_barrier_init(barrier_tfenc_feedforward_gelu, nullptr, params->num_th_batch);
	pthread_barrier_init(barrier_tfenc_feedforward_post, nullptr, params->num_th_batch);
	pthread_barrier_init(barrier_tfenc_feedforward_residual, nullptr, params->num_th_batch);
	pthread_barrier_init(barrier_tfenc_feedforward_layernorm, nullptr, params->num_th_batch);

	attention_layer = new self_attention_column(params, bert_state, layer_idx, is_training);
	attention_fc = new layer_dense(bert_state->weight[layer_idx][WEIGHT_ATTENTION],bert_state->weight[layer_idx][WEIGHT_ATTENTION_BIAS],
	                               is_training, seq_length, hidden_size, hidden_size);
	attention_norm = layer_norm(bert_state->weight[layer_idx][WEIGHT_ATTENTION_GAMMA], bert_state->weight[layer_idx][WEIGHT_ATTENTION_BETA],
	                            is_training, batch, seq_length, hidden_size);
	prev_feedforward = new layer_dense(bert_state->weight[layer_idx][WEIGHT_PREV_FFW], bert_state->weight[layer_idx][WEIGHT_PREV_FFB],
	                                   is_training, seq_length, feedforward_size, hidden_size);
	gelu_feedforward = gelu(is_training, batch, feedforward_size * seq_length);
	post_feedforward = new layer_dense(bert_state->weight[layer_idx][WEIGHT_POST_FFW], bert_state->weight[layer_idx][WEIGHT_POST_FFB],
	                                   is_training, seq_length, hidden_size, feedforward_size);
	feedforward_norm = layer_norm(bert_state->weight[layer_idx][WEIGHT_FF_GAMMA], bert_state->weight[layer_idx][WEIGHT_FF_BETA],
	                              is_training, batch, seq_length, hidden_size);
	context=(float **)mkl_calloc(batch, sizeof(float*), 64);
	attention_result=(float **)mkl_calloc(batch, sizeof(float*), 64);
	attention_residual=(float **)mkl_calloc(batch, sizeof(float*), 64);
	attention_layernorm=(float **)mkl_calloc(batch, sizeof(float*), 64);
	feedforward_intermediate=(float **)mkl_calloc(batch, sizeof(float*), 64);
	feedforward_gelu=(float**)mkl_calloc(batch, sizeof(float *), 64);
	feedforward_result=(float **)mkl_calloc(batch, sizeof(float *), 64);
	feedforward_residual=(float**)mkl_calloc(batch, sizeof(float*), 64);
	feedforward_layernorm=(float **)mkl_calloc(batch, sizeof(float*), 64);

	for(int i=0; i<batch; i++){
		attention_residual[i] = (float *)mkl_calloc(seq_length, hidden_size*sizeof(float), 64);
		feedforward_residual[i] = (float *)mkl_calloc(seq_length, hidden_size*sizeof(float), 64);
		attention_layernorm[i] = (float *)mkl_calloc(seq_length, hidden_size*sizeof(float), 64);
		attention_result[i] = (float*)mkl_calloc(seq_length, hidden_size*sizeof(float), 64);
		feedforward_layernorm[i] = (float *)mkl_calloc(seq_length, hidden_size*sizeof(float), 64);
		feedforward_intermediate[i] =(float *)mkl_calloc(seq_length, feedforward_size*sizeof(float*), 64);
		feedforward_result[i] =(float *)mkl_calloc(seq_length, hidden_size*sizeof(float*), 64);
		feedforward_gelu[i] =(float *)mkl_calloc(seq_length, feedforward_size*sizeof(float *), 64);

		memset(attention_residual[i],       0, sizeof(float) * seq_length * hidden_size);
		memset(feedforward_residual[i],     0, sizeof(float) * seq_length * hidden_size);
		memset(attention_layernorm[i],      0, sizeof(float) * seq_length * hidden_size);
		memset(attention_result[i],         0, sizeof(float) * seq_length * hidden_size);
		memset(feedforward_layernorm[i],    0, sizeof(float) * seq_length * hidden_size);
		memset(feedforward_result[i],       0, sizeof(float) * seq_length * hidden_size);
		memset(feedforward_intermediate[i], 0, sizeof(float) * seq_length * feedforward_size);
		memset(feedforward_gelu[i],         0, sizeof(float) * seq_length * feedforward_size);
	}
}

void transformer_encoder_column::dump_values() {
	printf("=== transformer_encoder_baseline ===\n");
	printf("=== [call] attention_layer->dump_values ===\n");
	attention_layer->dump_values();
	printf("=== [done] attention_layer->dump_values ===\n");
	for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
		printf("context[%d] => ", batch_idx);
		for (int i = 0; i < 16; i++) { printf("%f ", context[batch_idx][i]); } printf("\n");
		printf("attention_result[%d] => ", batch_idx);
		for (int i = 0; i < 16; i++) { printf("%f ", attention_result[batch_idx][i]); } printf("\n");
		printf("attention_residual[%d] => ", batch_idx);
		for (int i = 0; i < 16; i++) { printf("%f ", attention_residual[batch_idx][i]); } printf("\n");
		printf("attention_layernorm[%d] => ", batch_idx);
		for (int i = 0; i < 16; i++) { printf("%f ", attention_layernorm[batch_idx][i]); } printf("\n");
		printf("feedforward_intermediate[%d] => ", batch_idx);
		for (int i = 0; i < 16; i++) { printf("%f ", feedforward_intermediate[batch_idx][i]); } printf("\n");
		printf("feedforward_gelu[%d] => ", batch_idx);
		for (int i = 0; i < 16; i++) { printf("%f ", feedforward_gelu[batch_idx][i]); } printf("\n");
		printf("feedforward_result[%d] => ", batch_idx);
		for (int i = 0; i < 16; i++) { printf("%f ", feedforward_result[batch_idx][i]); } printf("\n");
		printf("feedforward_residual[%d] => ", batch_idx);
		for (int i = 0; i < 16; i++) { printf("%f ", feedforward_residual[batch_idx][i]); } printf("\n");
		printf("feedforward_layernorm[%d] => ", batch_idx);
		for (int i = 0; i < 16; i++) { printf("%f ", feedforward_layernorm[batch_idx][i]); } printf("\n");
	}
}

float *transformer_encoder_column::forward(int batch_tid, float *a, float *attention_mask, int batch_idx, int layer_idx, Logger *loggers) {
	Logger &logger = loggers[Logger::get_logger_idx(batch_idx, -1)];

	pthread_barrier_wait(barrier_tfenc_attention_layer);
	context[batch_idx] = attention_layer->forward(batch_tid, a, attention_mask, batch_idx, layer_idx, loggers);

	pthread_barrier_wait(barrier_tfenc_attention_fc);
	logger.infer_logging_begin(ATTENTION_FC, layer_idx);
	attention_fc->forward(context[batch_idx], CblasNoTrans, attention_result[batch_idx]);
	logger.infer_logging_end(ATTENTION_FC, layer_idx);


	pthread_barrier_wait(barrier_tfenc_attention_residual);
	logger.infer_logging_begin(ATTENTION_RESIDUAL_CONNECT, layer_idx);
	add(attention_result[batch_idx], a, seq_length * hidden_size, attention_residual[batch_idx]);
	logger.infer_logging_end(ATTENTION_RESIDUAL_CONNECT, layer_idx);


	pthread_barrier_wait(barrier_tfenc_attention_layernorm);
	logger.infer_logging_begin(ATTENTION_LAYERNORM, layer_idx);
	attention_norm.forward(attention_residual[batch_idx], batch_idx, attention_layernorm[batch_idx]);
	logger.infer_logging_end(ATTENTION_LAYERNORM, layer_idx);


	pthread_barrier_wait(barrier_tfenc_feedforward_pre);
	logger.infer_logging_begin(FEEDFORWARD_PRE, layer_idx);
	prev_feedforward->forward(attention_layernorm[batch_idx], CblasNoTrans, feedforward_intermediate[batch_idx]);
	logger.infer_logging_end(FEEDFORWARD_PRE, layer_idx);


	pthread_barrier_wait(barrier_tfenc_feedforward_gelu);
	logger.infer_logging_begin(FEEDFORWARD_GELU, layer_idx);
	gelu_feedforward.forward(feedforward_intermediate[batch_idx],batch_idx, feedforward_gelu[batch_idx]);
	logger.infer_logging_end(FEEDFORWARD_GELU, layer_idx);


	pthread_barrier_wait(barrier_tfenc_feedforward_post);
	logger.infer_logging_begin(FEEDFORWARD_POST, layer_idx);
	post_feedforward->forward(feedforward_gelu[batch_idx], CblasNoTrans, feedforward_result[batch_idx]);
	logger.infer_logging_end(FEEDFORWARD_POST, layer_idx);


	pthread_barrier_wait(barrier_tfenc_feedforward_residual);
	logger.infer_logging_begin(FEEDFORWARD_RESIDUAL_CONNECT, layer_idx);
	add(attention_layernorm[batch_idx], feedforward_result[batch_idx], seq_length * hidden_size, feedforward_residual[batch_idx]);
	logger.infer_logging_end(FEEDFORWARD_RESIDUAL_CONNECT, layer_idx);


	pthread_barrier_wait(barrier_tfenc_feedforward_layernorm);
	logger.infer_logging_begin(FEEDFORWARD_LAYERNORM, layer_idx);
	feedforward_norm.forward(feedforward_residual[batch_idx], batch_idx, feedforward_layernorm[batch_idx]);
	logger.infer_logging_end(FEEDFORWARD_LAYERNORM, layer_idx);

	return feedforward_layernorm[batch_idx];
}

void transformer_encoder_column::transformer_encoder_deinit() {
	delete(barrier_tfenc_attention_layer);
	delete(barrier_tfenc_attention_fc);
	delete(barrier_tfenc_attention_residual);
	delete(barrier_tfenc_attention_layernorm);
	delete(barrier_tfenc_feedforward_pre);
	delete(barrier_tfenc_feedforward_gelu);
	delete(barrier_tfenc_feedforward_post);
	delete(barrier_tfenc_feedforward_residual);
	delete(barrier_tfenc_feedforward_layernorm);

	attention_layer->self_attention_deinit();
	delete(attention_layer);
	delete(attention_fc);
	attention_norm.layer_norm_deinit();
	delete(prev_feedforward);
	gelu_feedforward.gelu_deinit();
	delete(post_feedforward);
	feedforward_norm.layer_norm_deinit();

	for(int i=0; i<batch; i++){
		mkl_free(attention_residual[i]);
		mkl_free(feedforward_residual[i]);
		mkl_free(attention_layernorm[i]);
		mkl_free(attention_result[i]);
		mkl_free(feedforward_layernorm[i]);
		mkl_free(feedforward_intermediate[i]);
		mkl_free(feedforward_result[i]);
		mkl_free(feedforward_gelu[i]);
	}
	mkl_free(attention_residual); attention_residual = nullptr;
	mkl_free(feedforward_residual); feedforward_residual = nullptr;
	mkl_free(attention_layernorm); attention_layernorm = nullptr;
	mkl_free(attention_result); attention_result = nullptr;
	mkl_free(feedforward_layernorm); feedforward_layernorm = nullptr;
	mkl_free(feedforward_intermediate); feedforward_intermediate = nullptr;
	mkl_free(feedforward_result); feedforward_result = nullptr;
	mkl_free(feedforward_gelu); feedforward_gelu = nullptr;
	mkl_free(context);
}

BERT::BERT(Params *_params, BERT_State *_bert_state, bool _is_training, bool use_onehot, bool use_token_type, bool use_position_embedding) {
	params = _params;
	bert_state = _bert_state;
	is_training = _is_training;
	batch_thread_num = params->num_th_batch;
	batch = bert_state->num_batch;
	num_layers = bert_state->num_layer;
	num_heads = bert_state->num_heads;
	head_thread_num = params->num_th_head;
	seq_length = bert_state->seq_length;
	hidden_size = bert_state->hidden_size;

	barrier_bert_embedding_lookup = new pthread_barrier_t;
	barrier_bert_embedding_postprocessor = new pthread_barrier_t;
	barrier_bert_transformer_layer = new pthread_barrier_t;
	pthread_barrier_init(barrier_bert_embedding_lookup, nullptr, batch_thread_num);
	pthread_barrier_init(barrier_bert_embedding_postprocessor, nullptr, batch_thread_num);
	pthread_barrier_init(barrier_bert_transformer_layer, nullptr, batch_thread_num);

	//embedding
	embedding_layer = new embedding(is_training, bert_state->emb_weight[0], bert_state->vocab_size, use_onehot,
	                                bert_state->emb_weight[1], bert_state->emb_weight[2], use_token_type, use_position_embedding,
	                                bert_state->emb_weight[3], bert_state->emb_weight[4], bert_state->token_size, bert_state->num_batch, bert_state->hidden_size, bert_state->seq_length);

	embedding_output_lookup=(float**)mkl_calloc(batch, sizeof(float*), 64);
	embedding_output=(float**)mkl_calloc(batch, sizeof(float*), 64);
	for(int i=0; i<batch; i++){
		embedding_output_lookup[i] = (float*)mkl_calloc(seq_length, hidden_size*sizeof(float), 64);
		embedding_output[i] = (float*)mkl_calloc(seq_length, hidden_size*sizeof(float), 64);
		memset(embedding_output_lookup[i], 0, sizeof(float) * seq_length * hidden_size);
		memset(embedding_output[i], 0, sizeof(float) * seq_length * hidden_size);
	}
	//encoder stack
	transformer_layer = (transformer_encoder_column**)mkl_malloc(num_layers * sizeof(transformer_encoder_column*), 64);

	for (int i = 0; i < num_layers; i++) {
		transformer_layer[i] = new transformer_encoder_column(params, bert_state, i, is_training);
	}

	//post embedding
	pooler = new layer_dense(bert_state->pooler_weight[0], bert_state->pooler_weight[1], is_training, batch, hidden_size, hidden_size);
	layer_output = (float ***)mkl_calloc(batch, sizeof(float **), 64);
	for(int i=0; i<batch; i++){
		layer_output[i]= (float **)mkl_calloc((num_layers+1), sizeof(float*), 64);
	}
	output = (float **)mkl_calloc(batch, sizeof(float*), 64);
	attention_mask = (float **)mkl_calloc(batch, sizeof(float*), 64);
	ones_for_attention_mask = (float **)mkl_malloc(batch * sizeof(float *), 64);
	for (int i = 0; i < batch; i++) {
		attention_mask[i] = (float*)mkl_calloc(seq_length, seq_length * sizeof(float), 64);
		ones_for_attention_mask[i] = (float*)mkl_malloc(seq_length * 1 * sizeof(float), 64);
		memset(attention_mask[i], 0, sizeof(float) * seq_length * seq_length);
		for (int j = 0; j < 1 * seq_length; j++) { ones_for_attention_mask[i][j] = 1; }
	}

	first_token_tensor = (float *)mkl_calloc(batch, hidden_size*sizeof(float), 64);
	pooler_output = (float *)mkl_calloc(hidden_size, hidden_size*sizeof(float), 64);
	memset(first_token_tensor, 0, sizeof(float) * batch * hidden_size);
	memset(pooler_output, 0, sizeof(float) * hidden_size * hidden_size);
}

void BERT::create_attention_mask_from_input(float **input_mask, MKL_INT q_seq_length, MKL_INT k_seq_length, int num_batch) {
	matmul mm = matmul(false, q_seq_length, k_seq_length, 1);
	mm.forward(ones_for_attention_mask, input_mask, false, attention_mask, num_batch);
}

void BERT::dump_values() {
	for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
		printf("=== layer_idx:%d ===\n",layer_idx);
		transformer_layer[layer_idx]->dump_values();
	}
}

float *BERT::forward(int **input_ids, float **input_mask, int **token_type_ids, Logger *loggers) {
	std::thread **th_list;
	pthread_t **att_th_list;

	create_attention_mask_from_input(input_mask, seq_length, seq_length, batch);    // set attention_mask

	Logger::init_csmon_arr(batch_thread_num, head_thread_num);
	th_list=(std::thread **)malloc(sizeof(std::thread*)*batch_thread_num);
	att_th_list = (pthread_t**)malloc(sizeof(pthread_t*) * batch_thread_num);

	for(int batch_tid = 0; batch_tid < batch_thread_num; batch_tid++) {
		th_list[batch_tid] = new std::thread([=]{forward_pth(batch_tid, batch_tid*batch/batch_thread_num, input_ids, attention_mask, token_type_ids, output, loggers);});

		att_th_list[batch_tid] = new pthread_t[head_thread_num];
		for (int head_tid = 0; head_tid < head_thread_num; head_tid++) {
			if (pthread_create(&att_th_list[batch_tid][head_tid], nullptr, attention_threads_column,
					bert_state->att_st_list[batch_tid * params->num_th_head + head_tid]) != 0) {
				assert(0);
			}

			cpu_set_t cpuset;
			CPU_ZERO(&cpuset);
			CPU_SET(batch_tid * params->num_th_head + head_tid, &cpuset);
			if (pthread_setaffinity_np(att_th_list[batch_tid][head_tid], sizeof(cpu_set_t), &cpuset) < 0) {
				assert(0);
			}
		}
	}
	for(int batch_tid = 0; batch_tid < batch_thread_num; batch_tid++) {
		for (int head_tid = 0; head_tid < head_thread_num; head_tid++) {
			if (pthread_join(att_th_list[batch_tid][head_tid], nullptr) != 0) {
				assert(0);
			}
		}
		delete[] (att_th_list[batch_tid]);
		th_list[batch_tid]->join();
		delete(th_list[batch_tid]);
	}
	free(att_th_list);
	free(th_list);

	squeeze(output, batch, hidden_size, first_token_tensor);
	pooler->forward(first_token_tensor, CblasNoTrans, pooler_output);
	return pooler_output;
}

void *BERT::forward_pth(int batch_tid, int id, int ** input, float ** mask, int ** token_type, float **t_output, Logger *loggers){
	int batch_idx;
	int logger_idx;
	mkl_set_num_threads_local(params->num_th_head);
	Logger::init_csmon(batch_tid, -1, true);		// count child threads (to track mkl threads)
	Logger::csmon_start(batch_tid, -1);

	for(int i=0; i<batch/batch_thread_num; i++){
		batch_idx = id + i;
		logger_idx = Logger::get_logger_idx(batch_idx, -1);
		loggers[logger_idx].set_tids(batch_tid, -1);
		loggers[logger_idx].set_cs_counter(Logger::get_csmon(batch_tid, -1)->get_counter());

		pthread_barrier_wait(barrier_bert_embedding_lookup);
		embedding_layer->embedding_lookup(input[batch_idx], batch_idx, embedding_output_lookup[batch_idx], &loggers[logger_idx]);

		pthread_barrier_wait(barrier_bert_embedding_postprocessor);
		embedding_layer->embedding_postprocessor(embedding_output_lookup[batch_idx], token_type[batch_idx],
		                                         batch_idx, embedding_output[batch_idx], &loggers[logger_idx]);
		layer_output[batch_idx][0] = embedding_output[batch_idx];

		pthread_barrier_wait(barrier_bert_transformer_layer);
		for (int j = 0; j < num_layers; j++) {
			layer_output[batch_idx][j+1] = transformer_layer[j]->forward(batch_tid, layer_output[batch_idx][j], mask[batch_idx], batch_idx, j, loggers);
		}
		t_output[batch_idx] = layer_output[batch_idx][num_layers];

		loggers[logger_idx].free_cs_counter();
	}

	Logger::csmon_stop(batch_tid, -1);
	Logger::deinit_csmon(batch_tid, -1);
	return nullptr;
}

void BERT::BERT_deinit() {
	delete(barrier_bert_embedding_lookup);
	delete(barrier_bert_embedding_postprocessor);
	delete(barrier_bert_transformer_layer);

	embedding_layer->embedding_deinit();
	delete(embedding_layer);
	for (int i = 0; i < num_layers; i++) {
		transformer_layer[i]->transformer_encoder_deinit();
		delete(transformer_layer[i]);
	}
	mkl_free(transformer_layer); transformer_layer = nullptr;
	delete pooler; pooler = nullptr;

	for (int i = 0; i < batch;  i++) {
		mkl_free(layer_output[i]);
		mkl_free(attention_mask[i]);
		mkl_free(ones_for_attention_mask[i]);
		mkl_free(embedding_output_lookup[i]);
		mkl_free(embedding_output[i]);
	}
	mkl_free(layer_output); layer_output = nullptr;
	mkl_free(attention_mask); attention_mask = nullptr;
	mkl_free(ones_for_attention_mask); ones_for_attention_mask = nullptr;
	mkl_free(embedding_output_lookup); embedding_output_lookup = nullptr;
	mkl_free(embedding_output); embedding_output = nullptr;

	mkl_free(output); output = nullptr;
	mkl_free(first_token_tensor); first_token_tensor = nullptr;
	mkl_free(pooler_output); pooler_output = nullptr;
}
