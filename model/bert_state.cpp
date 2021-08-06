#include <cassert>
#include <string>
#include <iostream>
#include <fstream>
#include <thread>

#include "utils.hpp"
#include "log.hpp"
#include "bert_state.hpp"
#include "c_matrix.hpp"

using namespace std;

static void _init_bert_state_veri(Params *params, BERT_State *bert_state);
static void _init_bert_state_perf(Params *params, BERT_State *bert_state);

void init_bert_state(Params *params, BERT_State *bert_state) {
	if (params->execution_mode == EXEC_MODE_VERIFICATION) {
		_init_bert_state_veri(params, bert_state);
	} else if (params->execution_mode == EXEC_MODE_PERF_TEST) {
		_init_bert_state_perf(params, bert_state);
	} else {
		assert(0);
	}

	if (params->column_based) {
		assert(params->chunk_size);
		bert_state->column_based = true;
		bert_state->chunk_size = params->chunk_size;
		bert_state->num_chunks = bert_state->seq_length / params->chunk_size;
	}

	// Init Attention_State used to communication link to attention workers
	bert_state->att_st_list = (Attention_State**)malloc(params->num_th_batch * params->num_th_head * sizeof(struct Attention_State*));
	bert_state->ffw_st_list = (Feedforward_State**)malloc(params->num_th_batch * params->num_th_head * sizeof(struct Feedforward_State*));
	for (int i = 0; i < params->num_th_batch * params->num_th_head; i++) {
		bert_state->att_st_list[i] = (Attention_State*)malloc(sizeof(Attention_State));
		bert_state->ffw_st_list[i] = (Feedforward_State*)malloc(sizeof(Feedforward_State));
		memset(bert_state->att_st_list[i], 0, sizeof(Attention_State));
		memset(bert_state->ffw_st_list[i], 0, sizeof(Feedforward_State));
	}

	// barrier (for attention workers' start and end points)
	bert_state->barrier_attention_start = new pthread_barrier_t;
	bert_state->barrier_attention_end = new pthread_barrier_t;
	bert_state->barrier_ffw_start = new pthread_barrier_t;
	bert_state->barrier_ffw_end = new pthread_barrier_t;
	pthread_barrier_init(bert_state->barrier_attention_start, nullptr, params->num_th_batch * (params->num_th_head + 1));
	pthread_barrier_init(bert_state->barrier_attention_end, nullptr, params->num_th_batch * (params->num_th_head + 1));
	pthread_barrier_init(bert_state->barrier_ffw_start, nullptr, params->num_th_batch * (params->num_th_head + 1));
	pthread_barrier_init(bert_state->barrier_ffw_end, nullptr, params->num_th_batch * (params->num_th_head + 1));

	for (int batch_tid = 0; batch_tid < params->num_th_batch; batch_tid++) {
		for (int head_tid = 0; head_tid < params->num_th_head; head_tid++) {
			/// For attention workers
			bert_state->att_st_list[batch_tid * params->num_th_head + head_tid]->exec_mode = params->execution_mode;
			bert_state->att_st_list[batch_tid * params->num_th_head + head_tid]->batch_tid = batch_tid;
			bert_state->att_st_list[batch_tid * params->num_th_head + head_tid]->head_tid = head_tid;

			bert_state->att_st_list[batch_tid * params->num_th_head + head_tid]->num_batchs = bert_state->num_batch;
			bert_state->att_st_list[batch_tid * params->num_th_head + head_tid]->batch_th_num = params->num_th_batch;
			bert_state->att_st_list[batch_tid * params->num_th_head + head_tid]->num_heads = bert_state->num_heads;
			bert_state->att_st_list[batch_tid * params->num_th_head + head_tid]->head_th_num = params->num_th_head;
			bert_state->att_st_list[batch_tid * params->num_th_head + head_tid]->num_layers = bert_state->num_layer;
			bert_state->att_st_list[batch_tid * params->num_th_head + head_tid]->column_based = bert_state->column_based;
			bert_state->att_st_list[batch_tid * params->num_th_head + head_tid]->num_chunks = bert_state->num_chunks;
			bert_state->att_st_list[batch_tid * params->num_th_head + head_tid]->chunk_size = bert_state->chunk_size;
			bert_state->att_st_list[batch_tid * params->num_th_head + head_tid]->seq_length = bert_state->seq_length;
			bert_state->att_st_list[batch_tid * params->num_th_head + head_tid]->hidden_size = bert_state->hidden_size;
			bert_state->att_st_list[batch_tid * params->num_th_head + head_tid]->head_size = bert_state->hidden_size / bert_state->num_heads;

			bert_state->att_st_list[batch_tid * params->num_th_head + head_tid]->barrier_attention_start = bert_state->barrier_attention_start;
			bert_state->att_st_list[batch_tid * params->num_th_head + head_tid]->barrier_attention_end = bert_state->barrier_attention_end;


			/// For feedforward workers
			bert_state->ffw_st_list[batch_tid * params->num_th_head + head_tid]->exec_mode = params->execution_mode;
			bert_state->ffw_st_list[batch_tid * params->num_th_head + head_tid]->batch_tid = batch_tid;
			bert_state->ffw_st_list[batch_tid * params->num_th_head + head_tid]->head_tid = head_tid;

			bert_state->ffw_st_list[batch_tid * params->num_th_head + head_tid]->num_batchs = bert_state->num_batch;
			bert_state->ffw_st_list[batch_tid * params->num_th_head + head_tid]->batch_th_num = params->num_th_batch;
			bert_state->ffw_st_list[batch_tid * params->num_th_head + head_tid]->num_heads = bert_state->num_heads;
			bert_state->ffw_st_list[batch_tid * params->num_th_head + head_tid]->head_th_num = params->num_th_head;
			bert_state->ffw_st_list[batch_tid * params->num_th_head + head_tid]->num_layers = bert_state->num_layer;
			bert_state->ffw_st_list[batch_tid * params->num_th_head + head_tid]->seq_length = bert_state->seq_length;
			bert_state->ffw_st_list[batch_tid * params->num_th_head + head_tid]->hidden_size = bert_state->hidden_size;
			bert_state->ffw_st_list[batch_tid * params->num_th_head + head_tid]->head_size = bert_state->hidden_size / bert_state->num_heads;
			bert_state->ffw_st_list[batch_tid * params->num_th_head + head_tid]->feedforward_size = bert_state->feedforwardsize;
			bert_state->ffw_st_list[batch_tid * params->num_th_head + head_tid]->ffw_chunk_size = bert_state->feedforwardsize / bert_state->num_heads;

			bert_state->ffw_st_list[batch_tid * params->num_th_head + head_tid]->barrier_ffw_start = bert_state->barrier_ffw_start;
			bert_state->ffw_st_list[batch_tid * params->num_th_head + head_tid]->barrier_ffw_end = bert_state->barrier_ffw_end;
		}
	}
}


////////////////////////////////////////////////////////////
////////////// BERT verification mode //////////////////////
////////////////////////////////////////////////////////////
#define VERI_VOCABSIZE 30522
#define VERI_HIDDENSIZE 1024
#define VERI_TOKENSIZE 2
#define VERI_POSITIONSIZE 512
#define VERI_LAYERNUM 24
#define VERI_FEEDFORWARDSIZE 4096
#define VERI_BATCH 8
#define VERI_SEQLENGTH 256
#define VERI_NUMHEADS 16

static float *_getr2(const char *filename, int s0, int s1) {
	ifstream read(filename);
	if(!read.is_open()){
		cout<<"can not read file ("<<filename<<")"<<endl;
		exit(0);
	}
	float *weight=(float *)mkl_malloc(sizeof(float)*s0*s1, 64);
	for(int i=0;i<s0;i++){
		for(int j=0;j<s1;j++){
			read>>weight[i*s1+j];
		}
	}
	//cout<<"read: "<<weight[0]<<", "<<weight[s0*s1-1]<<endl;
	read.close();
	return weight;
}

static float **_getr3(const char * filename, int s0, int s1, int s2) {
	ifstream read(filename);
	if(!read.is_open()){
		cout<<"can not read file ("<<filename<<")"<<endl;
		exit(0);
	}
	float **weight=(float **)mkl_malloc(sizeof(float *)*s0, 64);
	for(int i=0;i<s0;i++) {
		weight[i] = (float *) mkl_malloc(sizeof(float) * s1 * s2, 64);
	}

	for(int i=0;i<s0;i++){
		for(int j=0;j<s1;j++){
			for(int k=0;k<s2;k++){
				read>>weight[i][j*s2+k];
			}
		}
	}
	//cout<<"read: "<<weight[0][0]<<", "<<weight[s0-1][s1*s2-1]<<endl;
	read.close();
	return weight;
}

static int ** _getir3(const char * filename, int s0, int s1, int s2) {
	ifstream read(filename);
	if(!read.is_open()){
		cout<<"can not read file ("<<filename<<")"<<endl;
		exit(0);
	}

	int **weight=(int **)mkl_malloc(sizeof(int *)*s0, 64);
	for(int i=0;i<s0;i++) {
		weight[i] = (int *) mkl_malloc(sizeof(int) * s1 * s2, 64);
	}

	for(int i=0;i<s0;i++){
		for(int j=0;j<s1;j++){
			for(int k=0;k<s2;k++){
				read>>weight[i][j*s2+k];
			}
		}
	}
	//cout<<"read: "<<weight[0][0]<<", "<<weight[s0-1][s1*s2-1]<<endl;
	read.close();
	return weight;
}

static void _init_veri_embed_weight(Params *params, BERT_State *bert_state) {
	string path(params->dir_weight + "/bert:embeddings:");

	bert_state->emb_weight[0] = _getr2((path+"word_embeddings:0").c_str(), bert_state->vocab_size, bert_state->hidden_size);
	bert_state->emb_weight[1] = _getr2((path+"token_type_embeddings:0").c_str(), bert_state->token_size, bert_state->hidden_size);
	bert_state->emb_weight[2] = _getr2((path+"position_embeddings:0").c_str(), VERI_POSITIONSIZE, bert_state->hidden_size);
	bert_state->emb_weight[3] = _getr2((path+"LayerNorm:gamma:0").c_str(), 1, bert_state->hidden_size);
	bert_state->emb_weight[4] = _getr2((path+"LayerNorm:beta:0").c_str(), 1, bert_state->hidden_size);

	cout<<"reading embedding file done"<<endl;
}

static void _init_veri_weight_th(int i, Params *params, BERT_State *bert_state) {
	string path(params->dir_weight + "/bert:encoder:layer_");

	bert_state->weight[i][WEIGHT_QLW] = _getr2((path+to_string(i)+":attention:self:query:kernel:0").c_str(), bert_state->hidden_size,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_QLB] = _getr2((path+to_string(i)+":attention:self:query:bias:0").c_str(), 1,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_KLW] = _getr2((path+to_string(i)+":attention:self:key:kernel:0").c_str(), bert_state->hidden_size,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_KLB] = _getr2((path+to_string(i)+":attention:self:key:bias:0").c_str(), 1,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_VLW] = _getr2((path+to_string(i)+":attention:self:value:kernel:0").c_str(), bert_state->hidden_size,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_VLB] = _getr2((path+to_string(i)+":attention:self:value:bias:0").c_str(), 1,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_ATTENTION] = _getr2((path+to_string(i)+":attention:output:dense:kernel:0").c_str(), bert_state->hidden_size,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_ATTENTION_BIAS] = _getr2((path+to_string(i)+":attention:output:dense:bias:0").c_str(), 1,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_ATTENTION_GAMMA] = _getr2((path+to_string(i)+":attention:output:LayerNorm:gamma:0").c_str(), 1,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_ATTENTION_BETA] = _getr2((path+to_string(i)+":attention:output:LayerNorm:beta:0").c_str(), 1,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_PREV_FFW] = _getr2((path+to_string(i)+":intermediate:dense:kernel:0").c_str(), bert_state->hidden_size,bert_state->feedforwardsize);
	bert_state->weight[i][WEIGHT_PREV_FFB] = _getr2((path+to_string(i)+":intermediate:dense:bias:0").c_str(), 1,bert_state->feedforwardsize);
	bert_state->weight[i][WEIGHT_POST_FFW] = _getr2((path+to_string(i)+":output:dense:kernel:0").c_str(), bert_state->feedforwardsize,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_POST_FFB] = _getr2((path+to_string(i)+":output:dense:bias:0").c_str(), 1,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_FF_GAMMA] = _getr2((path+to_string(i)+":output:LayerNorm:gamma:0").c_str(), 1,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_FF_BETA] = _getr2((path+to_string(i)+":output:LayerNorm:beta:0").c_str(), 1,bert_state->hidden_size);
	cout<<"reading "<<to_string(i)<<"th layer weight file done"<<endl;
}

static void _init_veri_weight(Params *params, BERT_State *bert_state) {
	thread* th[bert_state->num_layer];
	for(int i=0;i<bert_state->num_layer;i++){
		th[i]=new thread(_init_veri_weight_th, i, params, bert_state);
	}
	for(int i=0;i<bert_state->num_layer;i++){
		th[i]->join();
	}
}

static void _init_pooler_weight(Params *params, BERT_State *bert_state){
	string path(params->dir_weight + "/bert:pooler:dense:");
	bert_state->pooler_weight[0] = _getr2((path+"kernel:0").c_str(), bert_state->hidden_size, bert_state->hidden_size);
	bert_state->pooler_weight[1] = _getr2((path+"bias:0").c_str(), 1, bert_state->hidden_size);
	cout<<"reading pooler file done"<<endl;
}

static void _init_bert_state_veri(Params *params, BERT_State *bert_state) {
	bert_state->vocab_size = VERI_VOCABSIZE;
	bert_state->token_size = VERI_TOKENSIZE;
	bert_state->num_batch = VERI_BATCH;
	bert_state->num_heads = VERI_NUMHEADS;
	bert_state->seq_length = VERI_SEQLENGTH;
	bert_state->hidden_size = VERI_HIDDENSIZE;
	bert_state->feedforwardsize = VERI_FEEDFORWARDSIZE;
	bert_state->num_layer = VERI_LAYERNUM;

	bert_state->emb_weight = (float **)malloc(sizeof(float*)*BERT_NUM_EMBED_WEIGHT);
	bert_state->weight = (float ***)malloc(sizeof(float **)*BERT_NUM_LAYER);
	bert_state->pooler_weight = (float **)malloc(sizeof(float*)*BERT_NUM_POOLER_WEIGHT);

	for(int i=0;i<bert_state->num_layer;i++)
		bert_state->weight[i] = (float **)malloc(sizeof(float*)*WEIGHT_MAX_NUM);

	_init_veri_embed_weight(params, bert_state);
	_init_veri_weight(params, bert_state);
	_init_pooler_weight(params, bert_state);
	cout<<"weight read done"<<endl;

	bert_state->input = _getir3((params->dir_smallest + "/embedding_lookup_input_ids0.txt").c_str(), bert_state->num_batch ,bert_state->seq_length, 1);
	bert_state->mask = _getr3((params->dir_smallest + "/to_mask_attention_mask2.txt").c_str(), bert_state->num_batch, bert_state->seq_length, 1);
	bert_state->token_type = _getir3((params->dir_smallest + "/token_type_ids1.txt").c_str(), bert_state->num_batch, bert_state->seq_length, 1);
	cout<<"input read done"<<endl;
}

float *load_pooled_output(Params *params, BERT_State *bert_state) {
	return _getr2((params->dir_smallest + "/pooled_output").c_str(), bert_state->num_batch, bert_state->hidden_size);
}

////////////////////////////////////////////////////////////
//////////////// BERT perf. test mode //////////////////////
////////////////////////////////////////////////////////////
static void _init_perf_read_chunks(FILE *fp, float *buf, int elem_num) {
	int rc;
	int size = elem_num * sizeof(float);
	int remaining = size;

	while (true) {
		if ((rc = fread((char*)buf + (size - remaining), 1, remaining, fp)) <= 0) {
			assert(0);
		} else {
			remaining -= rc;
			if (remaining > 0) {
				rewind(fp);
			} else {
				break;
			}
		}
	}
}

static void _init_perf_emb_weight(Params *params, BERT_State *bert_state) {
	char fname_buf[1024];
	snprintf(fname_buf, 1024, "%s/random_chunk%d.bin", params->dir_random_chunk, 0);
	FILE *fp_random_chunk = fopen(fname_buf, "rb");
	if (fp_random_chunk) {

		bert_state->emb_weight[EMB_WEIGHT_TABLE] = (float*) mkl_malloc(sizeof(float) * bert_state->vocab_size * bert_state->hidden_size, 64);
		bert_state->emb_weight[EMB_WEIGHT_TOKEN] = (float*) mkl_malloc(sizeof(float) * bert_state->token_size * bert_state->hidden_size, 64);
		bert_state->emb_weight[EMB_WEIGHT_POSITION] = (float*) mkl_malloc(sizeof(float) * bert_state->seq_length * bert_state->hidden_size, 64);
		bert_state->emb_weight[EMB_WEIGHT_GAMMA] = (float*) mkl_malloc(sizeof(float) * 1 * bert_state->hidden_size, 64);
		bert_state->emb_weight[EMB_WEIGHT_BETA] = (float*) mkl_malloc(sizeof(float) * 1 * bert_state->hidden_size, 64);

		_init_perf_read_chunks(fp_random_chunk, bert_state->emb_weight[EMB_WEIGHT_TABLE], bert_state->vocab_size * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->emb_weight[EMB_WEIGHT_TOKEN], bert_state->token_size * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->emb_weight[EMB_WEIGHT_POSITION], bert_state->seq_length * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->emb_weight[EMB_WEIGHT_GAMMA], 1 * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->emb_weight[EMB_WEIGHT_BETA], 1 * bert_state->hidden_size);

		fclose(fp_random_chunk);
	} else {
		bert_state->emb_weight[EMB_WEIGHT_TABLE] = (float*)mat_rand<float>(bert_state->vocab_size, bert_state->hidden_size);
		bert_state->emb_weight[EMB_WEIGHT_TOKEN] = mat_rand<float>(bert_state->token_size, bert_state->hidden_size);
		// Here, emb_weight[2] (position_table) has (VERI_POSITIONSIZE x hidden_size) dimension. However, we just simply
		// set its dimension as (seq_len x hidden_size) because copying it into position embedding (seq_len x hidden_size)
		// is the only usage.
		bert_state->emb_weight[EMB_WEIGHT_POSITION] = mat_rand<float>(bert_state->seq_length, bert_state->hidden_size);
		bert_state->emb_weight[EMB_WEIGHT_GAMMA] = mat_rand<float>(1, bert_state->hidden_size);
		bert_state->emb_weight[EMB_WEIGHT_BETA] = mat_rand<float>(1, bert_state->hidden_size);
	}
}

static void _init_perf_weight_th(int layer_idx, Params *params, BERT_State *bert_state) {
	char fname_buf[1024];
	snprintf(fname_buf, 1024, "%s/random_chunk%d.bin", params->dir_random_chunk, layer_idx);
	FILE *fp_random_chunk = fopen(fname_buf, "rb");
	if (fp_random_chunk) {
		bert_state->weight[layer_idx][WEIGHT_QLW] = (float*) mkl_malloc(sizeof(float) * bert_state->hidden_size * bert_state->hidden_size, 64);
		bert_state->weight[layer_idx][WEIGHT_QLB] = (float*) mkl_malloc(sizeof(float) * 1 * bert_state->hidden_size, 64);
		bert_state->weight[layer_idx][WEIGHT_KLW] = (float*) mkl_malloc(sizeof(float) * bert_state->hidden_size * bert_state->hidden_size, 64);
		bert_state->weight[layer_idx][WEIGHT_KLB] = (float*) mkl_malloc(sizeof(float) * 1 * bert_state->hidden_size, 64);
		bert_state->weight[layer_idx][WEIGHT_VLW] = (float*) mkl_malloc(sizeof(float) * bert_state->hidden_size * bert_state->hidden_size, 64);
		bert_state->weight[layer_idx][WEIGHT_VLB] = (float*) mkl_malloc(sizeof(float) * 1 * bert_state->hidden_size, 64);
		bert_state->weight[layer_idx][WEIGHT_ATTENTION] = (float*) mkl_malloc(sizeof(float) * bert_state->hidden_size * bert_state->hidden_size, 64);
		bert_state->weight[layer_idx][WEIGHT_ATTENTION_BIAS] = (float*) mkl_malloc(sizeof(float) * 1 * bert_state->hidden_size, 64);
		bert_state->weight[layer_idx][WEIGHT_ATTENTION_GAMMA] = (float*) mkl_malloc(sizeof(float) * 1 * bert_state->hidden_size, 64);
		bert_state->weight[layer_idx][WEIGHT_ATTENTION_BETA] = (float*) mkl_malloc(sizeof(float) * 1 * bert_state->hidden_size, 64);
		bert_state->weight[layer_idx][WEIGHT_PREV_FFW] = (float*) mkl_malloc(sizeof(float) * bert_state->hidden_size * bert_state->feedforwardsize, 64);
		bert_state->weight[layer_idx][WEIGHT_PREV_FFB] = (float*) mkl_malloc(sizeof(float) * 1 * bert_state->feedforwardsize, 64);
		bert_state->weight[layer_idx][WEIGHT_POST_FFW] = (float*) mkl_malloc(sizeof(float) * bert_state->feedforwardsize * bert_state->hidden_size, 64);
		bert_state->weight[layer_idx][WEIGHT_POST_FFB] = (float*) mkl_malloc(sizeof(float) * 1 * bert_state->hidden_size, 64);
		bert_state->weight[layer_idx][WEIGHT_FF_GAMMA] = (float*) mkl_malloc(sizeof(float) * 1 * bert_state->hidden_size, 64);
		bert_state->weight[layer_idx][WEIGHT_FF_BETA] = (float*) mkl_malloc(sizeof(float) * 1 * bert_state->hidden_size, 64);

		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_QLW], bert_state->hidden_size * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_QLB], 1 * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_KLW], bert_state->hidden_size * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_KLB], 1 * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_VLW], bert_state->hidden_size * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_VLB], 1 * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_ATTENTION], bert_state->hidden_size * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_ATTENTION_BIAS], 1 * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_ATTENTION_GAMMA], 1 * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_ATTENTION_BETA], 1 * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_PREV_FFW], bert_state->hidden_size * bert_state->feedforwardsize);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_PREV_FFB], 1 * bert_state->feedforwardsize);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_POST_FFW], bert_state->feedforwardsize * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_POST_FFB], 1 * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_FF_GAMMA], 1 * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_FF_BETA], 1 * bert_state->hidden_size);

		fclose(fp_random_chunk);
	} else {
		bert_state->weight[layer_idx][WEIGHT_QLW] = mat_rand<float>(bert_state->hidden_size, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_QLB] = mat_rand<float>(1, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_KLW] = mat_rand<float>(bert_state->hidden_size, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_KLB] = mat_rand<float>(1, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_VLW] = mat_rand<float>(bert_state->hidden_size, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_VLB] = mat_rand<float>(1, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_ATTENTION] = mat_rand<float>(bert_state->hidden_size, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_ATTENTION_BIAS] = mat_rand<float>(1, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_ATTENTION_GAMMA] = mat_rand<float>(1, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_ATTENTION_BETA] = mat_rand<float>(1, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_PREV_FFW] = mat_rand<float>(bert_state->hidden_size, bert_state->feedforwardsize);
		bert_state->weight[layer_idx][WEIGHT_PREV_FFB] = mat_rand<float>(1, bert_state->feedforwardsize);
		bert_state->weight[layer_idx][WEIGHT_POST_FFW] = mat_rand<float>(bert_state->feedforwardsize, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_POST_FFB] = mat_rand<float>(1, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_FF_GAMMA] = mat_rand<float>(1, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_FF_BETA] = mat_rand<float>(1, bert_state->hidden_size);
	}
}


static void _init_bert_state_perf(Params *params, BERT_State *bert_state) {
	bert_state->vocab_size = params->vocab_size;
	bert_state->token_size = params->token_size;
	bert_state->num_batch = params->num_batch;
	bert_state->num_heads = params->num_heads;
	bert_state->seq_length = params->seq_length;
	bert_state->hidden_size = params->hidden_size;
	bert_state->feedforwardsize = params->feedforwardsize;
	bert_state->num_layer = params->num_layer;

	bert_state->emb_weight = (float **)malloc(sizeof(float*)*BERT_NUM_EMBED_WEIGHT);
	bert_state->weight = (float ***)malloc(sizeof(float **)*bert_state->num_layer);
	for (int i = 0; i < bert_state->num_layer; i++) {
		bert_state->weight[i] = (float **) malloc(sizeof(float *) * WEIGHT_MAX_NUM);
	}
	bert_state->pooler_weight = (float **)malloc(sizeof(float*)*BERT_NUM_POOLER_WEIGHT);

	/// init embed_weight
	_init_perf_emb_weight(params, bert_state);
	mark_time("<Init_BERT_State_Perf> emb_weight END");

	/// init weight
	thread *th_weights[bert_state->num_layer];
	for (int layer_idx = 0; layer_idx < bert_state->num_layer; layer_idx++) {
		th_weights[layer_idx] = new thread(_init_perf_weight_th, layer_idx, params, bert_state);
	}
	for (int layer_idx = 0; layer_idx < bert_state->num_layer; layer_idx++) {
		th_weights[layer_idx]->join();
		delete(th_weights[layer_idx]);
	}
	mark_time("<Init_BERT_State_Perf> weight END");

	/// init pooler_weight
	bert_state->pooler_weight[0] = mat_rand<float>(bert_state->hidden_size, bert_state->hidden_size);
	bert_state->pooler_weight[1] = mat_rand<float>(1, bert_state->hidden_size);

	/// init input
	bert_state->input = (int**)mkl_malloc(sizeof(int *) * bert_state->num_batch, 64);
	bert_state->mask = (float**)mkl_malloc(sizeof(float *) * bert_state->num_batch, 64);
	bert_state->token_type = (int**)mkl_malloc(sizeof(int *) * bert_state->num_batch, 64);
	for (int i = 0; i < bert_state->num_batch; i++) {
		// maximum value of lookup_input_ids0.txt is 9287 minimum is 0
		bert_state->input[i] = mat_rand<int>(bert_state->seq_length, 1, 0, 10000);
		// all values of mask are 0 or 1.
		bert_state->mask[i] = mat_rand<float>(bert_state->seq_length, 1, 0, 1);
		for (int j = 0; j < bert_state->seq_length; j++) {
			bert_state->mask[i][j] = (bert_state->mask[i][j] > 0.5) ? 1.0 : 0.0;
		}
		// all values of mask are 0 or 1.
		bert_state->token_type[i] = mat_rand<int>(bert_state->seq_length, 1, 0, 1);
	}
	mark_time("<Init_BERT_State_Perf> pooler and input END");
}


void deinit_bert_state(Params *params, BERT_State *bert_state) {
	for (int i = 0; i < BERT_NUM_EMBED_WEIGHT; i++) {
		mkl_free(bert_state->emb_weight[i]);
		bert_state->emb_weight[i] = nullptr;
	}
	free(bert_state->emb_weight); bert_state->emb_weight = nullptr;

	for (int i = 0; i < bert_state->num_layer; i++) {
		for (int j = 0; j < WEIGHT_MAX_NUM; j++) {
			mkl_free(bert_state->weight[i][j]);
			bert_state->weight[i][j] = nullptr;
		}
		free(bert_state->weight[i]);
		bert_state->weight[i] = nullptr;
	}
	free(bert_state->weight); bert_state->weight = nullptr;

	for (int i = 0; i < BERT_NUM_POOLER_WEIGHT; i++) {
		mkl_free(bert_state->pooler_weight[i]);
		bert_state->pooler_weight[i] = nullptr;
	}
	free(bert_state->pooler_weight); bert_state->pooler_weight = nullptr;

	for (int i = 0; i < bert_state->num_batch; i++) {
		mkl_free(bert_state->input[i]);
		bert_state->input[i] = nullptr;
	}
	mkl_free(bert_state->input); bert_state->input = nullptr;

	for (int i = 0; i < bert_state->num_batch; i++) {
		mkl_free(bert_state->mask[i]);
		bert_state->mask[i] = nullptr;
	}
	mkl_free(bert_state->mask); bert_state->mask = nullptr;

	for (int i = 0; i < bert_state->num_batch; i++) {
		mkl_free(bert_state->token_type[i]);
		bert_state->token_type[i] = nullptr;
	}
	mkl_free(bert_state->token_type); bert_state->token_type = nullptr;

	for (int i = 0; i < params->num_th_batch * params->num_th_head; i++) {
		free(bert_state->att_st_list[i]);
		free(bert_state->ffw_st_list[i]);
	}
	free(bert_state->att_st_list);
	free(bert_state->ffw_st_list);

	delete(bert_state->barrier_attention_start);
	delete(bert_state->barrier_attention_end);
	delete(bert_state->barrier_ffw_start);
	delete(bert_state->barrier_ffw_end);
}
