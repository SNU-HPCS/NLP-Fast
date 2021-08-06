#include <iostream>
#include <cmath>
#include <sys/sysinfo.h>
#include <cstring>
#include <cassert>
#include <cuda_profiler_api.h>

#include "bert_state.hpp"
#include "utils.hpp"
#include "log.hpp"
#include "bert_op_cpu.hpp"
#include "cuda_init.cuh"

using namespace std;

int main(int argc, char *argv[]){
	int rc;
	float *ret = nullptr, *ref = nullptr;
	BERT_State *bert_state = nullptr;
	gpu_cuda_context_t gpu_context;
	Params params = {};
	LatCounter lat_counter;

	// parse parameters
	parse_args(argc, argv, &params);
	if (param_sanity_check(&params) != 0) {
		printf("[ERR] fail to param_sanity_check\n");
		return -1;
	}
	mark_time("<Parsing> END");

	// Init bert_state
	bert_state = (BERT_State*)malloc(sizeof(BERT_State));
	init_bert_state(&params, bert_state);
	assert(((bert_state->seq_length * bert_state->seq_length) % params.thread_block_size) == 0);
	mark_time("<Init_BERT_State> END");

	// Init objects for GPU execution
	memset(&gpu_context, 0, sizeof(gpu_cuda_context_t));
	if ((rc = cuda_init(&params, bert_state, &gpu_context)) != 0) {
		fprintf(stderr, "Fail to cuda_init\n");
		goto err;
	}
	mark_time("<Init_BERT_GPU> END");

	/// Start execution!!! (CPU Part) - Embedding
	create_attention_mask_from_input(bert_state);
	apply_embedding(bert_state);
	mark_time("<BERT> Embedding END");

	/// Start execution!!! (GPU Part) - Inference
	/// 1. cuda memory alloc
	if ((rc = cuda_mem_alloc(&params, bert_state, &gpu_context)) != 0) {
		fprintf(stderr, "Fail to cuda_mem_alloc\n");
		goto err;
	}

	/// 2. cuda memory value initialization
	if ((rc = cuda_mem_init(&params, bert_state, &gpu_context)) != 0) {
		fprintf(stderr, "Fail to cuda_mem_init\n");
		goto err;
	}
	mark_time("<Init_BERT_GPU> Memory & Value END");

	/// 2. Main GPU execution
//	cudaProfilerStart();
	lat_counter.begin();
	if ((rc = cuda_bert_main(&params, bert_state, &gpu_context)) != 0) {
		fprintf(stderr, "Fail to cuda_bert_main\n");
		goto err;
	}
	lat_counter.end();
//	cudaProfilerStop();
	mark_time("<BERT> END");
	sync_all_buf_to_host(bert_state, &gpu_context);

	/// Verification (verification mode only)
	if (params.execution_mode == EXEC_MODE_VERIFICATION) {
		float *first_token_tensor;
		float *pooler_output;
		float **output;

		first_token_tensor = (float *)mkl_calloc(bert_state->num_batch, bert_state->hidden_size*sizeof(float), 64);
		pooler_output = (float *)mkl_calloc(bert_state->hidden_size, bert_state->hidden_size*sizeof(float), 64);
		output = (float **)malloc(bert_state->num_batch * sizeof (float*));
		for (int b = 0; b < bert_state->num_batch; b++) {
			output[b] = (float *)mkl_calloc(bert_state->seq_length, bert_state->hidden_size*sizeof(float), 64);
			matcopy_col_to_row_float(output[b], gpu_context.h_buf_ffw_layernorm[b],
					bert_state->seq_length, bert_state->hidden_size);
		}
		memset(first_token_tensor, 0, sizeof(float) * bert_state->num_batch * bert_state->hidden_size);
		memset(pooler_output, 0, sizeof(float) * bert_state->hidden_size * bert_state->hidden_size);
		layer_dense *pooler = new layer_dense(bert_state->pooler_weight[0], bert_state->pooler_weight[1],
				false, bert_state->num_batch, bert_state->hidden_size, bert_state->hidden_size);

		squeeze(output, bert_state->num_batch, bert_state->hidden_size, first_token_tensor);
		pooler->forward(first_token_tensor, CblasNoTrans, pooler_output);

		/// Extract answer
		ref = load_pooled_output(&params, bert_state);

		/// Answer check
		cout<<"difference, "<<diff(pooler_output, ref, bert_state->num_batch * bert_state->hidden_size)<<endl;

		/// Free
		delete pooler;
		mkl_free(first_token_tensor);
		mkl_free(pooler_output);
		for (int b = 0; b < bert_state->num_batch; b++) {
			mkl_free(output[b]);
		}
		free(output);
	} else if (params.execution_mode == EXEC_MODE_PERF_TEST) {
		/// Dump latency & cachestat
		printf("==== Execution latency ====\n");
		printf("elapsed: %lf\n",lat_counter.get_latency());
	}

	// Deinit (bert, logger, bert_state)
	cuda_context_deinit(bert_state, &gpu_context);
	free(params.dir_random_chunk);
	deinit_bert_state(&params, bert_state);
	free(bert_state);

	return rc;

err:
	return rc;
}
