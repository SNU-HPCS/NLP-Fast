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
#include "cuda_multi_init.cuh"

using namespace std;

void *bert_multi_gpu_thread(void *ptr) {
	int rc;
	auto multi_gpu_thread_arg = (multi_gpu_thread_arg_t*)ptr;
	const int gpu_id = multi_gpu_thread_arg->gpu_id;
	gpu_cuda_context_t *gpu_context = &multi_gpu_thread_arg->gpu_contexts[gpu_id];
	Params *params = multi_gpu_thread_arg->params;
	BERT_State *bert_state = multi_gpu_thread_arg->bert_state;

	/// Set gpu_id
	if ((rc = set_device(gpu_id)) != 0) {
		fprintf(stderr, "<gpuid: %d> Fail to set_device\n", gpu_id); goto err;
	}

	/// cublas init
	if ((rc = cuda_multi_cublas_init(params, gpu_context, gpu_id)) != 0) {
		fprintf(stderr, "<gpuid: %d> Fail to cuda_multi_cublas_init\n", gpu_id); goto err;
	}

	/// Memory allocation
	if ((rc = cuda_multi_mem_alloc(params, bert_state, gpu_context)) != 0) {
		fprintf(stderr, "<gpuid: %d> Fail to cuda_multi_mem_alloc\n", gpu_id); goto err;
	}

	/// Memory initialization
	if ((rc = cuda_multi_mem_init(params, bert_state, gpu_context)) != 0) {
		fprintf(stderr, "<gpuid: %d> Fail to cuda_multi_mem_init\n", gpu_id); goto err;
	}

	/// Wait for init
	pthread_barrier_wait(multi_gpu_thread_arg->multi_gpu_barrier_total_start);

	/// Do execution
	if ((rc = cuda_multi_bert_main(multi_gpu_thread_arg)) != 0) {
		fprintf(stderr, "<gpuid: %d> Fail to cuda_multi_bert_main\n", gpu_id); goto err;
	}

	/// Wait for end
	pthread_barrier_wait(multi_gpu_thread_arg->multi_gpu_barrier_total_end);

	return nullptr;
err:
	assert(0);
}

int main(int argc, char *argv[]){
	int rc;
	float *ret = nullptr, *ref = nullptr;
	BERT_State *bert_state = nullptr;
	gpu_cuda_context_t *gpu_contexts;
	pthread_t *thread_arr;
	multi_gpu_thread_arg_t *thread_args;
	Params params = {};
	LatCounter lat_counter;
	int phy_num_gpus;
	pthread_barrier_t *barrier_total_start = new pthread_barrier_t;
	pthread_barrier_t *barrier_total_end = new pthread_barrier_t;
	pthread_barrier_t *multi_gpu_barrier_local_att_fc_rsum = new pthread_barrier_t;
	pthread_barrier_t *multi_gpu_barrier_local_att_fc_rsum_rescopy = new pthread_barrier_t;
	pthread_barrier_t *multi_gpu_barrier_local_att_fc_rsum_rescopy_done = new pthread_barrier_t;
	pthread_barrier_t *multi_gpu_barrier_local_ffw_rsum = new pthread_barrier_t;
	pthread_barrier_t *multi_gpu_barrier_local_ffw_rsum_rescopy = new pthread_barrier_t;
	pthread_barrier_t *multi_gpu_barrier_local_ffw_rsum_rescopy_done = new pthread_barrier_t;

	// parse parameters
	parse_args(argc, argv, &params);
	if (param_sanity_check(&params) != 0) {
		printf("[ERR] fail to param_sanity_check\n");
		return -1;
	}
	mark_time("<Parsing> END");

	/// Init bert_state
	bert_state = (BERT_State*)malloc(sizeof(BERT_State));
	init_bert_state(&params, bert_state);
	assert(((bert_state->seq_length * bert_state->seq_length) % params.thread_block_size) == 0);
	mark_time("<Init_BERT_State> END");

	/// GPU hardware # check
	if ((rc = get_device_count(&phy_num_gpus)) != 0) {
		fprintf(stderr, "Fail to get_device_count\n"); goto err;
	}
	if (params.num_gpus > phy_num_gpus) {
		fprintf(stderr, "Physical gpu_num(%d) is lower than a given num_gpu (%d)\n", phy_num_gpus, params.num_gpus); goto err;
	}

	/// Init objects for GPU execution (Host-side memory)
	gpu_contexts = (gpu_cuda_context_t*)malloc(sizeof(gpu_cuda_context_t) * params.num_gpus);
	memset(gpu_contexts, 0, sizeof(gpu_cuda_context_t) * params.num_gpus);
	if ((rc = cuda_multi_host_mem_alloc(&params, bert_state, &gpu_contexts[0])) != 0) {
		fprintf(stderr, "Fail to get_device_count\n"); goto err;
	}
	for (int i = 1; i < params.num_gpus; i++) {
		memcpy(&gpu_contexts[i], &gpu_contexts[0], sizeof(gpu_cuda_context_t));
	}
	mark_time("<Init_BERT_GPU> END");

	/// Start execution!!! (CPU Part) - Embedding
	create_attention_mask_from_input(bert_state);
	apply_embedding(bert_state);
	mark_time("<BERT> Embedding END");


	/// Init pthreads
	pthread_barrier_init(barrier_total_start, nullptr, params.num_gpus + 1);
	pthread_barrier_init(barrier_total_end, nullptr, params.num_gpus + 1);
	pthread_barrier_init(multi_gpu_barrier_local_att_fc_rsum, nullptr, params.num_gpus);
	pthread_barrier_init(multi_gpu_barrier_local_att_fc_rsum_rescopy, nullptr, params.num_gpus);
	pthread_barrier_init(multi_gpu_barrier_local_att_fc_rsum_rescopy_done, nullptr, params.num_gpus);
	pthread_barrier_init(multi_gpu_barrier_local_ffw_rsum, nullptr, params.num_gpus);
	pthread_barrier_init(multi_gpu_barrier_local_ffw_rsum_rescopy, nullptr, params.num_gpus);
	pthread_barrier_init(multi_gpu_barrier_local_ffw_rsum_rescopy_done, nullptr, params.num_gpus);

	thread_arr = (pthread_t*)malloc(sizeof(pthread_t) * params.num_gpus);
	thread_args = (multi_gpu_thread_arg_t*)malloc(sizeof(multi_gpu_thread_arg_t) * params.num_gpus);
	for (int i = 0; i < params.num_gpus; i++) {
		thread_args[i].gpu_num = params.num_gpus;
		thread_args[i].gpu_id = i;
		thread_args[i].params = &params;
		thread_args[i].bert_state = bert_state;
		thread_args[i].gpu_contexts = gpu_contexts;

		thread_args[i].multi_gpu_barrier_total_start = barrier_total_start;
		thread_args[i].multi_gpu_barrier_total_end = barrier_total_end;
		thread_args[i].multi_gpu_barrier_local_att_fc_rsum = multi_gpu_barrier_local_att_fc_rsum;
		thread_args[i].multi_gpu_barrier_local_att_fc_rsum_rescopy = multi_gpu_barrier_local_att_fc_rsum_rescopy;
		thread_args[i].multi_gpu_barrier_local_att_fc_rsum_rescopy_done = multi_gpu_barrier_local_att_fc_rsum_rescopy_done;
		thread_args[i].multi_gpu_barrier_local_ffw_rsum = multi_gpu_barrier_local_ffw_rsum;
		thread_args[i].multi_gpu_barrier_local_ffw_rsum_rescopy = multi_gpu_barrier_local_ffw_rsum_rescopy;
		thread_args[i].multi_gpu_barrier_local_ffw_rsum_rescopy_done = multi_gpu_barrier_local_ffw_rsum_rescopy_done;

		if (pthread_create(&thread_arr[i], nullptr, bert_multi_gpu_thread, (void *)&thread_args[i]) < 0) {
			perror("thread create error:");
			assert(0);
		}
	}


	pthread_barrier_wait(barrier_total_start);
	lat_counter.begin();
	pthread_barrier_wait(barrier_total_end);
	lat_counter.end();
	mark_time("<BERT> END");

	/// Verification (verification mode only)
	if (params.execution_mode == EXEC_MODE_VERIFICATION) {
		float *first_token_tensor;
		float *pooler_output;
		float **output;

		first_token_tensor = (float *)mkl_calloc(bert_state->num_batch, bert_state->hidden_size*sizeof(float), 64);
		pooler_output = (float *)mkl_calloc(bert_state->hidden_size, bert_state->hidden_size*sizeof(float), 64);
		output = (float **)malloc(bert_state->num_batch * sizeof (float*));
		sync_all_buf_to_host(bert_state, &gpu_contexts[0]);
		for (int b = 0; b < bert_state->num_batch; b++) {
			output[b] = (float *)mkl_calloc(bert_state->seq_length, bert_state->hidden_size*sizeof(float), 64);
			matcopy_col_to_row_float(output[b], gpu_contexts[0].h_buf_ffw_layernorm[b],
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
	cuda_multi_host_context_deinit(bert_state, &gpu_contexts[0], params.num_gpus);
	for (int i = 0; i < params.num_gpus; i++) {
		cuda_multi_dev_context_deinit(bert_state, &gpu_contexts[i]);
	}
	free(params.dir_random_chunk);
	deinit_bert_state(&params, bert_state);
	free(bert_state);
	free(gpu_contexts);
	free(thread_arr);

	delete(barrier_total_start);
	delete(barrier_total_end);
	delete(multi_gpu_barrier_local_att_fc_rsum);
	delete(multi_gpu_barrier_local_att_fc_rsum_rescopy);
	delete(multi_gpu_barrier_local_att_fc_rsum_rescopy_done);
	delete(multi_gpu_barrier_local_ffw_rsum);
	delete(multi_gpu_barrier_local_ffw_rsum_rescopy);
	delete(multi_gpu_barrier_local_ffw_rsum_rescopy_done);

	return rc;

err:
	return rc;
}
