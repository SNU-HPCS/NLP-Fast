#include <cassert>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <thread>
#include <mkl.h>

#include "utils.hpp"
#include "log.hpp"
#include "bert_state.hpp"

using namespace std;

static void _init_bert_state_veri(Params *params, BERT_State *bert_state);
static void _init_bert_state_perf(Params *params, BERT_State *bert_state);

static void init_att_fc_weight_splitted(BERT_State *bert_state) {
	const int num_layer = bert_state->num_layer;
	const int num_heads = bert_state->num_heads;
	const int seq_length = bert_state->seq_length;
	const int hidden_size = bert_state->hidden_size;
	const int head_size = bert_state->hidden_size / bert_state->num_heads;

	bert_state->weight_attention_fc_splitted = (float ***)malloc(num_layer * sizeof(float**));
	bert_state->weight_attention_fc_bias_splitted = (float **)malloc(num_layer * sizeof(float*));
	for (int layer_idx = 0; layer_idx < num_layer; layer_idx++) {
		bert_state->weight_attention_fc_splitted[layer_idx] = (float **)malloc(num_heads * sizeof(float*));
		for (int head_idx = 0; head_idx < num_heads; head_idx++) {
			bert_state->weight_attention_fc_splitted[layer_idx][head_idx] = (float *)malloc(head_size * hidden_size * sizeof(float));
		}
		bert_state->weight_attention_fc_bias_splitted[layer_idx] = (float *)malloc(seq_length* hidden_size * sizeof(float));
	}

	for (int layer_idx = 0; layer_idx < num_layer; layer_idx++) {
		for (int head_idx = 0; head_idx < num_heads; head_idx++) {
			memcpy(bert_state->weight_attention_fc_splitted[layer_idx][head_idx],
					&bert_state->weight[layer_idx][WEIGHT_ATTENTION][head_idx * head_size * hidden_size],
					head_size * hidden_size * sizeof(float));
		}
		memcpy(bert_state->weight_attention_fc_bias_splitted[layer_idx],
				bert_state->weight[layer_idx][WEIGHT_ATTENTION_BIAS],
				seq_length * hidden_size * sizeof(float));
	}
}

static void init_ffw_weight_splitted(BERT_State *bert_state) {
	const int num_layer = bert_state->num_layer;
	const int num_ffw_chunks = bert_state->num_heads;
	const int seq_length = bert_state->seq_length;
	const int hidden_size = bert_state->hidden_size;
	const int feedforward_size = bert_state->feedforwardsize;
	const int ffw_chunk_size = bert_state->feedforwardsize / bert_state->num_heads;

	bert_state->weight_ffw_prev_splitted = (float ***)malloc(num_layer * sizeof(float**));
	bert_state->weight_ffw_prev_bias_splitted = (float ***)malloc(num_layer * sizeof(float**));
	bert_state->weight_ffw_post_splitted = (float ***)malloc(num_layer * sizeof(float**));
	bert_state->weight_ffw_post_bias_splitted = (float **)malloc(num_layer * sizeof(float*));

	for (int layer_idx = 0; layer_idx < num_layer; layer_idx++) {
		bert_state->weight_ffw_prev_splitted[layer_idx] = (float **)malloc(num_ffw_chunks * sizeof(float*));
		bert_state->weight_ffw_prev_bias_splitted[layer_idx] = (float **)malloc(num_ffw_chunks * sizeof(float*));
		bert_state->weight_ffw_post_splitted[layer_idx] = (float **)malloc(num_ffw_chunks * sizeof(float*));

		for (int ffw_chunk_idx = 0; ffw_chunk_idx < num_ffw_chunks; ffw_chunk_idx++) {
			bert_state->weight_ffw_prev_splitted[layer_idx][ffw_chunk_idx] = (float *)malloc(hidden_size * ffw_chunk_size * sizeof(float));
			bert_state->weight_ffw_prev_bias_splitted[layer_idx][ffw_chunk_idx] = (float *)malloc(seq_length * ffw_chunk_size * sizeof(float));
			bert_state->weight_ffw_post_splitted[layer_idx][ffw_chunk_idx] = (float *)malloc(ffw_chunk_size * hidden_size * sizeof(float));
		}
		bert_state->weight_ffw_post_bias_splitted[layer_idx] = (float *)malloc(seq_length * hidden_size * sizeof(float));
	}

	for (int layer_idx = 0; layer_idx < num_layer; layer_idx++) {
		for (int ffw_chunk_idx = 0; ffw_chunk_idx < num_ffw_chunks; ffw_chunk_idx++) {
			for (int k = 0; k < hidden_size; k++) {
				memcpy(&bert_state->weight_ffw_prev_splitted[layer_idx][ffw_chunk_idx][k * ffw_chunk_size],
						&bert_state->weight[layer_idx][WEIGHT_PREV_FFW][k * feedforward_size + ffw_chunk_idx * ffw_chunk_size],
						ffw_chunk_size * sizeof(float));
			}
			for (int s = 0; s < seq_length; s++) {
				memcpy(&bert_state->weight_ffw_prev_bias_splitted[layer_idx][ffw_chunk_idx][s * ffw_chunk_size],
						&bert_state->weight[layer_idx][WEIGHT_PREV_FFB][s * feedforward_size + ffw_chunk_idx * ffw_chunk_size],
						ffw_chunk_size * sizeof(float));
			}

			memcpy(bert_state->weight_ffw_post_splitted[layer_idx][ffw_chunk_idx],
					&bert_state->weight[layer_idx][WEIGHT_POST_FFW][ffw_chunk_idx * ffw_chunk_size * hidden_size],
					ffw_chunk_size * hidden_size * sizeof(float));
		}
		memcpy(bert_state->weight_ffw_post_bias_splitted[layer_idx],
				bert_state->weight[layer_idx][WEIGHT_POST_FFB],
				seq_length * hidden_size * sizeof(float));
	}
}

void init_bert_state(Params *params, BERT_State *bert_state) {
	if (params->execution_mode == EXEC_MODE_VERIFICATION) {
		_init_bert_state_veri(params, bert_state);
	} else if (params->execution_mode == EXEC_MODE_PERF_TEST) {
		_init_bert_state_perf(params, bert_state);
	} else {
		assert(0);
	}
	init_att_fc_weight_splitted(bert_state);
	init_ffw_weight_splitted(bert_state);

	/// Init intermeidate values
	bert_state->ones_for_attention_mask = (float **)mkl_calloc(bert_state->num_batch, sizeof(float *), 64);
	bert_state->attention_mask = (float **)mkl_calloc(bert_state->num_batch, sizeof(float*), 64);
	bert_state->m_attention_mask = (float **)mkl_calloc(bert_state->num_batch, sizeof(float*), 64);
	for (int i = 0; i < bert_state->num_batch; i++) {
		bert_state->ones_for_attention_mask[i] = (float*)mkl_calloc(bert_state->seq_length, sizeof(float), 64);
		bert_state->attention_mask[i] = (float*)mkl_calloc(bert_state->seq_length, bert_state->seq_length * sizeof(float), 64);
		bert_state->m_attention_mask[i] = (float*)mkl_calloc(bert_state->seq_length, bert_state->seq_length * sizeof(float), 64);
		for (int j = 0; j < bert_state->seq_length; j++) { bert_state->ones_for_attention_mask[i][j] = 1; }
		memset(bert_state->attention_mask[i], 0, sizeof(float) * bert_state->seq_length * bert_state->seq_length);
		memset(bert_state->m_attention_mask[i], 0, sizeof(float) * bert_state->seq_length * bert_state->seq_length);
	}

	bert_state->embedding_output=(float**)mkl_calloc(bert_state->num_batch, sizeof(float*), 64);
	for(int i=0; i<bert_state->num_batch; i++){
		bert_state->embedding_output[i] = (float*)mkl_calloc(bert_state->seq_length, bert_state->hidden_size*sizeof(float), 64);
		memset(bert_state->embedding_output[i], 0, sizeof(float) * bert_state->seq_length * bert_state->hidden_size);
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
	float *weight = nullptr;
	ifstream read(filename);
	if(!read.is_open()){
		cout<<"can not read file ("<<filename<<")"<<endl;
		exit(0);
	}
	weight=(float *)malloc(sizeof(float)*s0*s1);
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
	float **weight = nullptr;
	ifstream read(filename);
	if(!read.is_open()){
		cout<<"can not read file ("<<filename<<")"<<endl;
		exit(0);
	}
	weight=(float **)malloc(sizeof(float *)*s0);
	for(int i=0;i<s0;i++) {
		weight[i] = (float *) malloc(sizeof(float) * s1 * s2);
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

static int **_getir3(const char * filename, int s0, int s1, int s2) {
	int **weight;
	ifstream read(filename);
	if(!read.is_open()){
		cout<<"can not read file ("<<filename<<")"<<endl;
		exit(0);
	}

	weight = (int **)malloc(sizeof(int *)*s0);
	for(int i=0;i<s0;i++) {
		weight[i] = (int *) malloc(sizeof(int) * s1 * s2);
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
	bert_state->weight[i][WEIGHT_QLB] = vec_to_mat_span_float(bert_state->weight[i][WEIGHT_QLB], bert_state->seq_length, bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_KLW] = _getr2((path+to_string(i)+":attention:self:key:kernel:0").c_str(), bert_state->hidden_size,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_KLB] = _getr2((path+to_string(i)+":attention:self:key:bias:0").c_str(), 1,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_KLB] = vec_to_mat_span_float(bert_state->weight[i][WEIGHT_KLB], bert_state->seq_length, bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_VLW] = _getr2((path+to_string(i)+":attention:self:value:kernel:0").c_str(), bert_state->hidden_size,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_VLB] = _getr2((path+to_string(i)+":attention:self:value:bias:0").c_str(), 1,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_VLB] = vec_to_mat_span_float(bert_state->weight[i][WEIGHT_VLB], bert_state->seq_length, bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_ATTENTION] = _getr2((path+to_string(i)+":attention:output:dense:kernel:0").c_str(), bert_state->hidden_size,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_ATTENTION_BIAS] = _getr2((path+to_string(i)+":attention:output:dense:bias:0").c_str(), 1,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_ATTENTION_BIAS] = vec_to_mat_span_float(bert_state->weight[i][WEIGHT_ATTENTION_BIAS], bert_state->seq_length, bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_ATTENTION_GAMMA] = _getr2((path+to_string(i)+":attention:output:LayerNorm:gamma:0").c_str(), 1,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_ATTENTION_GAMMA] = vec_to_mat_span_float(bert_state->weight[i][WEIGHT_ATTENTION_GAMMA], bert_state->seq_length, bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_ATTENTION_BETA] = _getr2((path+to_string(i)+":attention:output:LayerNorm:beta:0").c_str(), 1,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_ATTENTION_BETA] = vec_to_mat_span_float(bert_state->weight[i][WEIGHT_ATTENTION_BETA], bert_state->seq_length, bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_PREV_FFW] = _getr2((path+to_string(i)+":intermediate:dense:kernel:0").c_str(), bert_state->hidden_size,bert_state->feedforwardsize);
	bert_state->weight[i][WEIGHT_PREV_FFB] = _getr2((path+to_string(i)+":intermediate:dense:bias:0").c_str(), bert_state->seq_length, bert_state->feedforwardsize);
	bert_state->weight[i][WEIGHT_PREV_FFB] = vec_to_mat_span_float(bert_state->weight[i][WEIGHT_PREV_FFB], bert_state->seq_length, bert_state->feedforwardsize);
	bert_state->weight[i][WEIGHT_POST_FFW] = _getr2((path+to_string(i)+":output:dense:kernel:0").c_str(), bert_state->feedforwardsize,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_POST_FFB] = _getr2((path+to_string(i)+":output:dense:bias:0").c_str(), bert_state->seq_length, bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_POST_FFB] = vec_to_mat_span_float(bert_state->weight[i][WEIGHT_POST_FFB], bert_state->seq_length, bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_FF_GAMMA] = _getr2((path+to_string(i)+":output:LayerNorm:gamma:0").c_str(), 1,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_FF_GAMMA] = vec_to_mat_span_float(bert_state->weight[i][WEIGHT_FF_GAMMA], bert_state->seq_length, bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_FF_BETA] = _getr2((path+to_string(i)+":output:LayerNorm:beta:0").c_str(), 1,bert_state->hidden_size);
	bert_state->weight[i][WEIGHT_FF_BETA] = vec_to_mat_span_float(bert_state->weight[i][WEIGHT_FF_BETA], bert_state->seq_length, bert_state->hidden_size);
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

	for(int i=0;i<bert_state->num_layer;i++) {
		bert_state->weight[i] = (float **) malloc(sizeof(float *) * WEIGHT_MAX_NUM);
	}

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

		bert_state->emb_weight[EMB_WEIGHT_TABLE] = (float*) malloc(sizeof(float) * bert_state->vocab_size * bert_state->hidden_size);
		bert_state->emb_weight[EMB_WEIGHT_TOKEN] = (float*) malloc(sizeof(float) * bert_state->token_size * bert_state->hidden_size);
		bert_state->emb_weight[EMB_WEIGHT_POSITION] = (float*) malloc(sizeof(float) * bert_state->seq_length * bert_state->hidden_size);
		bert_state->emb_weight[EMB_WEIGHT_GAMMA] = (float*) malloc(sizeof(float) * 1 * bert_state->hidden_size);
		bert_state->emb_weight[EMB_WEIGHT_BETA] = (float*) malloc(sizeof(float) * 1 * bert_state->hidden_size);

		_init_perf_read_chunks(fp_random_chunk, bert_state->emb_weight[EMB_WEIGHT_TABLE], bert_state->vocab_size * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->emb_weight[EMB_WEIGHT_TOKEN], bert_state->token_size * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->emb_weight[EMB_WEIGHT_POSITION], bert_state->seq_length * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->emb_weight[EMB_WEIGHT_GAMMA], 1 * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->emb_weight[EMB_WEIGHT_BETA], 1 * bert_state->hidden_size);

		fclose(fp_random_chunk);
	} else {
		bert_state->emb_weight[EMB_WEIGHT_TABLE] = mat_rand_float(bert_state->vocab_size, bert_state->hidden_size);
		bert_state->emb_weight[EMB_WEIGHT_TOKEN] = mat_rand_float(bert_state->token_size, bert_state->hidden_size);
		// Here, emb_weight[2] (position_table) has (VERI_POSITIONSIZE x hidden_size) dimension. However, we just simply
		// set its dimension as (seq_len x hidden_size) because copying it into position embedding (seq_len x hidden_size)
		// is the only usage.
		bert_state->emb_weight[EMB_WEIGHT_POSITION] = mat_rand_float(bert_state->seq_length, bert_state->hidden_size);
		bert_state->emb_weight[EMB_WEIGHT_GAMMA] = mat_rand_float(1, bert_state->hidden_size);
		bert_state->emb_weight[EMB_WEIGHT_BETA] = mat_rand_float(1, bert_state->hidden_size);
	}
}

static void _init_perf_weight_th(int layer_idx, Params *params, BERT_State *bert_state) {
	char fname_buf[1024];
	snprintf(fname_buf, 1024, "%s/random_chunk%d.bin", params->dir_random_chunk, layer_idx);
	FILE *fp_random_chunk = fopen(fname_buf, "rb");
	if (fp_random_chunk) {
		bert_state->weight[layer_idx][WEIGHT_QLW] = (float*) malloc(sizeof(float) * bert_state->hidden_size * bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_QLB] = (float*) malloc(sizeof(float) * bert_state->seq_length * bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_KLW] = (float*) malloc(sizeof(float) * bert_state->hidden_size * bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_KLB] = (float*) malloc(sizeof(float) * bert_state->seq_length * bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_VLW] = (float*) malloc(sizeof(float) * bert_state->hidden_size * bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_VLB] = (float*) malloc(sizeof(float) * bert_state->seq_length * bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_ATTENTION] = (float*) malloc(sizeof(float) * bert_state->hidden_size * bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_ATTENTION_BIAS] = (float*) malloc(sizeof(float) * bert_state->seq_length * bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_ATTENTION_GAMMA] = (float*) malloc(sizeof(float) * bert_state->seq_length * bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_ATTENTION_BETA] = (float*) malloc(sizeof(float) * bert_state->seq_length * bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_PREV_FFW] = (float*) malloc(sizeof(float) * bert_state->hidden_size * bert_state->feedforwardsize);
		bert_state->weight[layer_idx][WEIGHT_PREV_FFB] = (float*) malloc(sizeof(float) * bert_state->seq_length * bert_state->feedforwardsize);
		bert_state->weight[layer_idx][WEIGHT_POST_FFW] = (float*) malloc(sizeof(float) * bert_state->feedforwardsize * bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_POST_FFB] = (float*) malloc(sizeof(float) * bert_state->seq_length * bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_FF_GAMMA] = (float*) malloc(sizeof(float) * bert_state->seq_length * bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_FF_BETA] = (float*) malloc(sizeof(float) * bert_state->seq_length * bert_state->hidden_size);

		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_QLW], bert_state->hidden_size * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_QLB], bert_state->seq_length * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_KLW], bert_state->hidden_size * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_KLB], bert_state->seq_length * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_VLW], bert_state->hidden_size * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_VLB], bert_state->seq_length * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_ATTENTION], bert_state->hidden_size * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_ATTENTION_BIAS], bert_state->seq_length * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_ATTENTION_GAMMA], bert_state->seq_length * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_ATTENTION_BETA], bert_state->seq_length * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_PREV_FFW], bert_state->hidden_size * bert_state->feedforwardsize);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_PREV_FFB], bert_state->seq_length * bert_state->feedforwardsize);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_POST_FFW], bert_state->feedforwardsize * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_POST_FFB], bert_state->seq_length * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_FF_GAMMA], bert_state->seq_length * bert_state->hidden_size);
		_init_perf_read_chunks(fp_random_chunk, bert_state->weight[layer_idx][WEIGHT_FF_BETA], bert_state->seq_length * bert_state->hidden_size);

		fclose(fp_random_chunk);
	} else {
		bert_state->weight[layer_idx][WEIGHT_QLW] = mat_rand_float(bert_state->hidden_size, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_QLB] = mat_rand_float(bert_state->seq_length, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_KLW] = mat_rand_float(bert_state->hidden_size, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_KLB] = mat_rand_float(bert_state->seq_length, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_VLW] = mat_rand_float(bert_state->hidden_size, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_VLB] = mat_rand_float(bert_state->seq_length, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_ATTENTION] = mat_rand_float(bert_state->hidden_size, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_ATTENTION_BIAS] = mat_rand_float(bert_state->seq_length, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_ATTENTION_GAMMA] = mat_rand_float(bert_state->seq_length, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_ATTENTION_BETA] = mat_rand_float(bert_state->seq_length, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_PREV_FFW] = mat_rand_float(bert_state->hidden_size, bert_state->feedforwardsize);
		bert_state->weight[layer_idx][WEIGHT_PREV_FFB] = mat_rand_float(bert_state->seq_length, bert_state->feedforwardsize);
		bert_state->weight[layer_idx][WEIGHT_POST_FFW] = mat_rand_float(bert_state->feedforwardsize, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_POST_FFB] = mat_rand_float(bert_state->seq_length, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_FF_GAMMA] = mat_rand_float(bert_state->seq_length, bert_state->hidden_size);
		bert_state->weight[layer_idx][WEIGHT_FF_BETA] = mat_rand_float(bert_state->seq_length, bert_state->hidden_size);
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
	bert_state->pooler_weight[0] = mat_rand_float(bert_state->hidden_size, bert_state->hidden_size);
	bert_state->pooler_weight[1] = mat_rand_float(1, bert_state->hidden_size);

	/// init input
	bert_state->input = (int**)malloc(sizeof(int *) * bert_state->num_batch);
	bert_state->mask = (float**)malloc(sizeof(float *) * bert_state->num_batch);
	bert_state->token_type = (int**)malloc(sizeof(int *) * bert_state->num_batch);
	for (int i = 0; i < bert_state->num_batch; i++) {
		// maximum value of lookup_input_ids0.txt is 9287 minimum is 0
		bert_state->input[i] = mat_rand_int(bert_state->seq_length, 1, 0, 10000);
		// all values of mask are 0 or 1.
		bert_state->mask[i] = mat_rand_float(bert_state->seq_length, 1, 0, 1);
		for (int j = 0; j < bert_state->seq_length; j++) {
			bert_state->mask[i][j] = (bert_state->mask[i][j] > 0.5) ? 1.0 : 0.0;
		}
		// all values of mask are 0 or 1.
		bert_state->token_type[i] = mat_rand_int(bert_state->seq_length, 1, 0, 1);
	}
	mark_time("<Init_BERT_State_Perf> pooler and input END");
}


void deinit_bert_state(Params *params, BERT_State *bert_state) {
	for (int i = 0; i < BERT_NUM_EMBED_WEIGHT; i++) {
		free(bert_state->emb_weight[i]);
		bert_state->emb_weight[i] = nullptr;
	}
	free(bert_state->emb_weight); bert_state->emb_weight = nullptr;

	for (int i = 0; i < bert_state->num_layer; i++) {
		for (int j = 0; j < WEIGHT_MAX_NUM; j++) {
			free(bert_state->weight[i][j]);
			bert_state->weight[i][j] = nullptr;
		}
		free(bert_state->weight[i]);
		bert_state->weight[i] = nullptr;

		for (int j = 0; j < bert_state->num_heads; j++) {
			free(bert_state->weight_attention_fc_splitted[i][j]);
			free(bert_state->weight_ffw_prev_splitted[i][j]);
			free(bert_state->weight_ffw_prev_bias_splitted[i][j]);
			free(bert_state->weight_ffw_post_splitted[i][j]);
		}
		free(bert_state->weight_attention_fc_splitted[i]);
		free(bert_state->weight_attention_fc_bias_splitted[i]);
		free(bert_state->weight_ffw_prev_splitted[i]);
		free(bert_state->weight_ffw_prev_bias_splitted[i]);
		free(bert_state->weight_ffw_post_splitted[i]);
		free(bert_state->weight_ffw_post_bias_splitted[i]);
	}
	free(bert_state->weight); bert_state->weight = nullptr;
	free(bert_state->weight_attention_fc_splitted); bert_state->weight_attention_fc_splitted = nullptr;
	free(bert_state->weight_attention_fc_bias_splitted); bert_state->weight_attention_fc_bias_splitted = nullptr;
	free(bert_state->weight_ffw_prev_splitted); bert_state->weight_ffw_prev_splitted = nullptr;
	free(bert_state->weight_ffw_prev_bias_splitted); bert_state->weight_ffw_prev_bias_splitted = nullptr;
	free(bert_state->weight_ffw_post_splitted); bert_state->weight_ffw_post_splitted = nullptr;
	free(bert_state->weight_ffw_post_bias_splitted); bert_state->weight_ffw_post_bias_splitted = nullptr;

	for (int i = 0; i < BERT_NUM_POOLER_WEIGHT; i++) {
		free(bert_state->pooler_weight[i]);
		bert_state->pooler_weight[i] = nullptr;
	}
	free(bert_state->pooler_weight); bert_state->pooler_weight = nullptr;

	for (int i = 0; i < bert_state->num_batch; i++) {
		free(bert_state->input[i]);
		bert_state->input[i] = nullptr;
	}
	free(bert_state->input); bert_state->input = nullptr;

	for (int i = 0; i < bert_state->num_batch; i++) {
		free(bert_state->mask[i]);
		bert_state->mask[i] = nullptr;
	}
	free(bert_state->mask); bert_state->mask = nullptr;

	for (int i = 0; i < bert_state->num_batch; i++) {
		free(bert_state->token_type[i]);
		bert_state->token_type[i] = nullptr;
	}
	free(bert_state->token_type); bert_state->token_type = nullptr;

	for (int i = 0 ; i < bert_state->num_batch; i++) {
		mkl_free(bert_state->ones_for_attention_mask[i]);
		mkl_free(bert_state->attention_mask[i]);
		mkl_free(bert_state->m_attention_mask[i]);
		mkl_free(bert_state->embedding_output[i]);
	}
	mkl_free(bert_state->ones_for_attention_mask);
	mkl_free(bert_state->attention_mask);
	mkl_free(bert_state->m_attention_mask);
	mkl_free(bert_state->embedding_output);
}
