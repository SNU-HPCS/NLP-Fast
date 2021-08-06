#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "tensor_op_gpu.cuh"
#include "cuda_multi_init.cuh"
#include "bert_state.hpp"

int get_device_count(int *num_gpus) {
	cudaError_t cuda_rc;
	if ((cuda_rc = cudaGetDeviceCount(num_gpus)) != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceCount (cuda_rc: %d)\n", cuda_rc);
		return -1;
	}
	return 0;
}

int set_device(int gpu_id) {
	cudaError_t cuda_rc;
	if ((cuda_rc = cudaSetDevice(gpu_id)) != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice (cuda_rc: %d)\n", cuda_rc);
		return -1;
	}
	if (gpu_id != 0) {
		if ((cuda_rc = cudaDeviceEnablePeerAccess(0, 0)) != cudaSuccess) {
			fprintf(stderr, "cudaDeviceDisablePeerAccess (%d -> %d) (cuda_rc: %d)\n", gpu_id, 0, cuda_rc);
			return -1;
		}
		cudaSetDevice(0);
		if ((cuda_rc = cudaDeviceEnablePeerAccess(gpu_id, 0)) != cudaSuccess) {
			fprintf(stderr, "cudaDeviceDisablePeerAccess (%d -> %d) (cuda_rc: %d)\n", 0, gpu_id, cuda_rc);
			return -1;
		}
		cudaSetDevice(gpu_id);
	}

	return 0;
}


int cuda_multi_cublas_init(Params *params, gpu_cuda_context_t *gpu_context, int gpu_id) {
	cublasStatus_t cublas_rc;

	/// used for p2p memcpy
//	if (gpu_id == 0) {
//		gpu_context->streams = (cudaStream_t *) malloc(sizeof(cudaStream_t) * params->num_gpus);
//		for (int t_gpu_idx = 0; t_gpu_idx < params->num_gpus; t_gpu_idx++) {
//			cudaStreamCreateWithFlags(&gpu_context->streams[t_gpu_idx], cudaStreamNonBlocking);
//		}
//	}

	gpu_context->cublas_handles = (cublasHandle_t*)malloc(sizeof(cublasHandle_t) * 1);
	if ((cublas_rc = cublasCreate(gpu_context->cublas_handles)) != CUBLAS_STATUS_SUCCESS)  {
		fprintf(stderr, "(i:%d) cublasCreate (cublas_rc: %d)\n", 0, cublas_rc);
		goto err;
	}

	return 0;
err:
	return -1;
}

int cuda_multi_host_mem_alloc(Params *params, BERT_State *bert_state, gpu_cuda_context_t *gpu_context) {
	const int num_batch = bert_state->num_batch;
	const int num_layer = bert_state->num_layer;
	const int num_heads = bert_state->num_heads;
	const int seq_length = bert_state->seq_length;
	const int hidden_size = bert_state->hidden_size;
	const int head_size = hidden_size / num_heads;
	const int feedforward_size = bert_state->feedforwardsize;
	const int ffw_chunk_size = feedforward_size / num_heads;
	cudaError_t cuda_rc;

	/// Onevec & onemat
//	if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_onevec, 1 * hidden_size * sizeof(float))) != cudaSuccess) {
//		fprintf(stderr, "cudaMallocHost (h_onevec) (cuda_rc: %d)\n", cuda_rc); goto err;
//	}
//	if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_onemat, hidden_size * hidden_size * sizeof(float))) != cudaSuccess) {
//		fprintf(stderr, "cudaMallocHost (h_onemat) (cuda_rc: %d)\n", cuda_rc); goto err;
//	}


	/// attention_mask
	gpu_context->h_attention_mask = (float **)malloc(num_batch * sizeof(float*));
	for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_attention_mask[batch_idx], seq_length * seq_length * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_attention_mask) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
	}

	/// Input
	gpu_context->h_input = (float**)malloc(num_batch * sizeof(float*));
	for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_input[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_input) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
	}

	/// Weight
	gpu_context->h_weight = (float ***)malloc(num_layer * sizeof(float**));
	gpu_context->h_weight_attention_fc_splitted = (float **)malloc(num_layer * sizeof(float*));
	gpu_context->h_weight_attention_fc_bias_splitted = (float **)malloc(num_layer * sizeof(float*));
	gpu_context->h_weight_ffw_prev = (float **)malloc(num_layer * sizeof(float*));
	gpu_context->h_weight_ffw_prev_bias = (float **)malloc(num_layer * sizeof(float*));
	gpu_context->h_weight_ffw_post_splitted = (float **)malloc(num_layer * sizeof(float*));
	gpu_context->h_weight_ffw_post_bias_splitted = (float **)malloc(num_layer * sizeof(float*));
	for (int layer_idx = 0; layer_idx < num_layer; layer_idx++) {
		gpu_context->h_weight[layer_idx] = (float **)malloc(WEIGHT_MAX_NUM * sizeof(float*));
		memset(gpu_context->h_weight[layer_idx], 0, WEIGHT_MAX_NUM * sizeof(float*));

		/// Host memory
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight[layer_idx][WEIGHT_QLW], hidden_size * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_weight[WEIGHT_QLW]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight[layer_idx][WEIGHT_QLB], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_weight[WEIGHT_QLB]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight[layer_idx][WEIGHT_KLW], hidden_size * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_weight[WEIGHT_KLW]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight[layer_idx][WEIGHT_KLB], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_weight[WEIGHT_KLB]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight[layer_idx][WEIGHT_VLW], hidden_size * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_weight[WEIGHT_VLW]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight[layer_idx][WEIGHT_VLB], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_weight[WEIGHT_VLB]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
//		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight[layer_idx][WEIGHT_ATTENTION], hidden_size * hidden_size * sizeof(float))) != cudaSuccess) {
//			fprintf(stderr, "cudaMallocHost (h_weight[WEIGHT_ATTENTION]) (cuda_rc: %d)\n", cuda_rc); goto err;
//		}
//		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight[layer_idx][WEIGHT_ATTENTION_BIAS], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
//			fprintf(stderr, "cudaMallocHost (h_weight[WEIGHT_ATTENTION_BIAS]) (cuda_rc: %d)\n", cuda_rc); goto err;
//		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight[layer_idx][WEIGHT_ATTENTION_GAMMA], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_weight[WEIGHT_ATTENTION_GAMMA]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight[layer_idx][WEIGHT_ATTENTION_BETA], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_weight[WEIGHT_ATTENTION_BETA]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
//		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight[layer_idx][WEIGHT_PREV_FFW], hidden_size * feedforward_size * sizeof(float))) != cudaSuccess) {
//			fprintf(stderr, "cudaMallocHost (h_weight[WEIGHT_PREV_FFW]) (cuda_rc: %d)\n", cuda_rc); goto err;
//		}
//		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight[layer_idx][WEIGHT_PREV_FFB], seq_length * feedforward_size * sizeof(float))) != cudaSuccess) {
//			fprintf(stderr, "cudaMallocHost (h_weight[WEIGHT_PREV_FFB]) (cuda_rc: %d)\n", cuda_rc); goto err;
//		}
//		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight[layer_idx][WEIGHT_POST_FFW], feedforward_size * hidden_size * sizeof(float))) != cudaSuccess) {
//			fprintf(stderr, "cudaMallocHost (h_weight[WEIGHT_POST_FFW]) (cuda_rc: %d)\n", cuda_rc); goto err;
//		}
//		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight[layer_idx][WEIGHT_POST_FFB], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
//			fprintf(stderr, "cudaMallocHost (h_weight[WEIGHT_POST_FFB]) (cuda_rc: %d)\n", cuda_rc); goto err;
//		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight[layer_idx][WEIGHT_FF_GAMMA], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_weight[WEIGHT_FF_GAMMA]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight[layer_idx][WEIGHT_FF_BETA], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_weight[WEIGHT_FF_BETA]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight_attention_fc_splitted[layer_idx], num_heads * head_size * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_weight_attention_fc_splitted) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight_attention_fc_bias_splitted[layer_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_weight_attention_fc_bias_splitted) (cuda_rc: %d)\n", cuda_rc); goto err;
		}

		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight_ffw_prev[layer_idx], hidden_size * feedforward_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_weight_ffw_prev) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight_ffw_prev_bias[layer_idx], seq_length * feedforward_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_weight_ffw_prev_bias) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight_ffw_post_splitted[layer_idx], num_heads * ffw_chunk_size * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_weight_ffw_post_splitted) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_weight_ffw_post_bias_splitted[layer_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_weight_ffw_post_bias_splitted) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
	}

	/// Intermediate buffer (Attention)
	gpu_context->h_buf_query   = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->h_buf_key     = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->h_buf_value   = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->h_buf_score   = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->h_buf_expsum  = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->h_buf_softmax = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->h_buf_att     = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->h_buf_att_fc_result_split = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->h_buf_att_layernorm = (float **)malloc(num_batch * sizeof(float**));
	gpu_context->h_buf_ffw_intermediate = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->h_buf_ffw_gelu = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->h_buf_ffw_result_split = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->h_buf_ffw_layernorm = (float **)malloc(num_batch * sizeof(float**));
//	gpu_context->h_buf_layernorm_mean = (float **)malloc(num_batch * sizeof(float*));
//	gpu_context->h_buf_layernorm_tmp = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->h_buf_layernorm_nrm_v = (float **)malloc(num_batch * sizeof(float*));
	for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_buf_query[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_buf_query) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_buf_key[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_buf_key) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_buf_value[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_buf_value) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_buf_score[batch_idx], num_heads * seq_length * seq_length * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_buf_score) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_buf_expsum[batch_idx], num_heads * seq_length * 1 * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_buf_expsum) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_buf_softmax[batch_idx], num_heads * seq_length * seq_length * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_buf_softmax) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_buf_att_fc_result_split[batch_idx], num_heads * seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_buf_att_fc_result_split) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_buf_ffw_intermediate[batch_idx], seq_length * feedforward_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_buf_ffw_intermediate) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_buf_ffw_gelu[batch_idx], seq_length * feedforward_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_buf_ffw_gelu) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_buf_ffw_result_split[batch_idx], num_heads * seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_buf_ffw_result_split) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}

		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_buf_att[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_buf_att) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_buf_att_layernorm[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_buf_att_layernorm) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_buf_ffw_layernorm[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_buf_ffw_layernorm) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
//		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_buf_layernorm_mean[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
//			fprintf(stderr, "cudaMallocHost (h_buf_layernorm_mean) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
//		}
//		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_buf_layernorm_tmp[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
//			fprintf(stderr, "cudaMallocHost (h_buf_layernorm_tmp) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
//		}
		if ((cuda_rc = cudaMallocHost((void**)&gpu_context->h_buf_layernorm_nrm_v[batch_idx], seq_length * 1 * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost (h_buf_layernorm_nrm_v) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
	}

	return 0;
err:
	return -1;
}

int cuda_multi_mem_alloc(Params *params, BERT_State *bert_state, gpu_cuda_context_t *gpu_context) {
	const int num_batch = bert_state->num_batch;
	const int num_layer = bert_state->num_layer;
	const int num_heads = bert_state->num_heads;
	const int seq_length = bert_state->seq_length;
	const int hidden_size = bert_state->hidden_size;
	const int head_size = hidden_size / num_heads;
	const int feedforward_size = bert_state->feedforwardsize;
	const int ffw_chunk_size = feedforward_size / num_heads;
	cudaError_t cuda_rc;

	/// Onevec & onemat
//	if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_onevec, 1 * hidden_size * sizeof(float))) != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc (d_onevec) (cuda_rc: %d)\n", cuda_rc); goto err;
//	}
//	if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_onemat, hidden_size * hidden_size * sizeof(float))) != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc (d_onemat) (cuda_rc: %d)\n", cuda_rc); goto err;
//	}


	/// attention_mask
	gpu_context->d_attention_mask = (float **)malloc(num_batch * sizeof(float*));
	for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_attention_mask[batch_idx], seq_length * seq_length * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_attention_mask) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
	}

	/// Input
	gpu_context->d_input = (float**)malloc(num_batch * sizeof(float*));
	for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_input[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_input) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
	}

	/// Weight
	gpu_context->d_weight = (float ***)malloc(num_layer * sizeof(float**));
	gpu_context->d_weight_attention_fc_splitted = (float **)malloc(num_layer * sizeof(float*));
	gpu_context->d_weight_attention_fc_bias_splitted = (float **)malloc(num_layer * sizeof(float*));
	gpu_context->d_weight_ffw_prev = (float **)malloc(num_layer * sizeof(float*));
	gpu_context->d_weight_ffw_prev_bias = (float **)malloc(num_layer * sizeof(float*));
	gpu_context->d_weight_ffw_post_splitted = (float **)malloc(num_layer * sizeof(float*));
	gpu_context->d_weight_ffw_post_bias_splitted = (float **)malloc(num_layer * sizeof(float*));
	for (int layer_idx = 0; layer_idx < num_layer; layer_idx++) {
		gpu_context->d_weight[layer_idx] = (float **)malloc(WEIGHT_MAX_NUM * sizeof(float*));
		memset(gpu_context->d_weight[layer_idx], 0, WEIGHT_MAX_NUM * sizeof(float*));

		/// GPU memory
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight[layer_idx][WEIGHT_QLW], hidden_size * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_weight[WEIGHT_QLW]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight[layer_idx][WEIGHT_QLB], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_weight[WEIGHT_QLB]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight[layer_idx][WEIGHT_KLW], hidden_size * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_weight[WEIGHT_KLW]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight[layer_idx][WEIGHT_KLB], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_weight[WEIGHT_KLB]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight[layer_idx][WEIGHT_VLW], hidden_size * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_weight[WEIGHT_VLW]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight[layer_idx][WEIGHT_VLB], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_weight[WEIGHT_VLB]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
//		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight[layer_idx][WEIGHT_ATTENTION], hidden_size * hidden_size * sizeof(float))) != cudaSuccess) {
//			fprintf(stderr, "cudaMalloc (d_weight[WEIGHT_ATTENTION]) (cuda_rc: %d)\n", cuda_rc); goto err;
//		}
//		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight[layer_idx][WEIGHT_ATTENTION_BIAS], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
//			fprintf(stderr, "cudaMalloc (d_weight[WEIGHT_ATTENTION_BIAS]) (cuda_rc: %d)\n", cuda_rc); goto err;
//		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight[layer_idx][WEIGHT_ATTENTION_GAMMA], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_weight[WEIGHT_ATTENTION_GAMMA]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight[layer_idx][WEIGHT_ATTENTION_BETA], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_weight[WEIGHT_ATTENTION_BETA]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
//		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight[layer_idx][WEIGHT_PREV_FFW], hidden_size * feedforward_size * sizeof(float))) != cudaSuccess) {
//			fprintf(stderr, "cudaMalloc (d_weight[WEIGHT_PREV_FFW]) (cuda_rc: %d)\n", cuda_rc); goto err;
//		}
//		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight[layer_idx][WEIGHT_PREV_FFB], seq_length * feedforward_size * sizeof(float))) != cudaSuccess) {
//			fprintf(stderr, "cudaMalloc (d_weight[WEIGHT_PREV_FFB]) (cuda_rc: %d)\n", cuda_rc); goto err;
//		}
//		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight[layer_idx][WEIGHT_POST_FFW], feedforward_size * hidden_size * sizeof(float))) != cudaSuccess) {
//			fprintf(stderr, "cudaMalloc (d_weight[WEIGHT_POST_FFW]) (cuda_rc: %d)\n", cuda_rc); goto err;
//		}
//		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight[layer_idx][WEIGHT_POST_FFB], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
//			fprintf(stderr, "cudaMalloc (d_weight[WEIGHT_POST_FFB]) (cuda_rc: %d)\n", cuda_rc); goto err;
//		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight[layer_idx][WEIGHT_FF_GAMMA], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_weight[WEIGHT_FF_GAMMA]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight[layer_idx][WEIGHT_FF_BETA], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_weight[WEIGHT_FF_BETA]) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight_attention_fc_splitted[layer_idx], num_heads * head_size * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_weight_attention_fc_splitted) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight_attention_fc_bias_splitted[layer_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_weight_attention_fc_bias_splitted) (cuda_rc: %d)\n", cuda_rc); goto err;
		}

		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight_ffw_prev[layer_idx], hidden_size * feedforward_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_weight_ffw_prev) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight_ffw_prev_bias[layer_idx], seq_length * feedforward_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_weight_ffw_prev_bias) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight_ffw_post_splitted[layer_idx], num_heads * ffw_chunk_size * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_weight_ffw_post_splitted) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_weight_ffw_post_bias_splitted[layer_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_weight_ffw_post_bias_splitted) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
	}

	/// Intermediate buffer (Attention)
	gpu_context->d_buf_query   = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->d_buf_key     = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->d_buf_value   = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->d_buf_score   = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->d_buf_expsum  = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->d_buf_softmax = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->d_buf_att     = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->d_buf_att_fc_result_split = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->d_buf_att_layernorm = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->d_buf_ffw_intermediate = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->d_buf_ffw_gelu = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->d_buf_ffw_result_split = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->d_buf_ffw_layernorm = (float **)malloc(num_batch * sizeof(float*));
//	gpu_context->d_buf_layernorm_mean = (float **)malloc(num_batch * sizeof(float*));
//	gpu_context->d_buf_layernorm_tmp = (float **)malloc(num_batch * sizeof(float*));
	gpu_context->d_buf_layernorm_nrm_v = (float **)malloc(num_batch * sizeof(float*));
	for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_buf_query[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_buf_query) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_buf_key[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_buf_key) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_buf_value[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_buf_value) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_buf_score[batch_idx], num_heads * seq_length * seq_length * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_buf_score) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_buf_expsum[batch_idx], num_heads * seq_length * 1 * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_buf_expsum) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_buf_softmax[batch_idx], num_heads * seq_length * seq_length * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_buf_softmax) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_buf_att_fc_result_split[batch_idx], num_heads * seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_buf_att_fc_result_split) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_buf_ffw_intermediate[batch_idx], seq_length * feedforward_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_buf_ffw_intermediate) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_buf_ffw_gelu[batch_idx], seq_length * feedforward_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_buf_ffw_gelu) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_buf_ffw_result_split[batch_idx], num_heads * seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_buf_ffw_result_split) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}

		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_buf_att[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_buf_att) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_buf_att_layernorm[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_buf_att_layernorm) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_buf_ffw_layernorm[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_buf_ffw_layernorm) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
//		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_buf_layernorm_mean[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
//			fprintf(stderr, "cudaMalloc (d_buf_layernorm_mean) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
//		}
//		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_buf_layernorm_tmp[batch_idx], seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
//			fprintf(stderr, "cudaMalloc (d_buf_layernorm_tmp) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
//		}
		if ((cuda_rc = cudaMalloc((void**)&gpu_context->d_buf_layernorm_nrm_v[batch_idx], seq_length * 1 * sizeof(float))) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (d_buf_layernorm_nrm_v) (cuda_rc: %d) (reason: %s)\n", cuda_rc, cudaGetErrorString(cuda_rc)); goto err;
		}
	}

	return 0;
err:
	return -1;
}

int cuda_multi_mem_init(Params *params, BERT_State *bert_state, gpu_cuda_context_t *gpu_context) {
	const int num_batch = bert_state->num_batch;
	const int num_layer = bert_state->num_layer;
	const int seq_length = bert_state->seq_length;
	const int hidden_size = bert_state->hidden_size;
	const int head_size = bert_state->hidden_size / bert_state->num_heads;
	const int ffw_chunk_size = bert_state->feedforwardsize / bert_state->num_heads;
	cudaError_t cuda_rc;

	/// Setting host memory
	if (params->execution_mode == EXEC_MODE_VERIFICATION) {
		/// Attention Mask
		for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
			matcopy_row_to_col_float(gpu_context->h_attention_mask[batch_idx], bert_state->m_attention_mask[batch_idx], seq_length, seq_length);
		}

		/// Input
		for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
			matcopy_row_to_col_float(gpu_context->h_input[batch_idx], bert_state->embedding_output[batch_idx], seq_length, hidden_size);
		}

		/// Weight
		for (int layer_idx = 0; layer_idx < num_layer; layer_idx++) {
			matcopy_row_to_col_float(gpu_context->h_weight[layer_idx][WEIGHT_QLW], bert_state->weight[layer_idx][WEIGHT_QLW], hidden_size, hidden_size);
			matcopy_row_to_col_float(gpu_context->h_weight[layer_idx][WEIGHT_QLB], bert_state->weight[layer_idx][WEIGHT_QLB], seq_length, hidden_size);
			matcopy_row_to_col_float(gpu_context->h_weight[layer_idx][WEIGHT_KLW], bert_state->weight[layer_idx][WEIGHT_KLW], hidden_size, hidden_size);
			matcopy_row_to_col_float(gpu_context->h_weight[layer_idx][WEIGHT_KLB], bert_state->weight[layer_idx][WEIGHT_KLB], seq_length, hidden_size);
			matcopy_row_to_col_float(gpu_context->h_weight[layer_idx][WEIGHT_VLW], bert_state->weight[layer_idx][WEIGHT_VLW], hidden_size, hidden_size);
			matcopy_row_to_col_float(gpu_context->h_weight[layer_idx][WEIGHT_VLB], bert_state->weight[layer_idx][WEIGHT_VLB], seq_length, hidden_size);
//			matcopy_row_to_col_float(gpu_context->h_weight[layer_idx][WEIGHT_ATTENTION], bert_state->weight[layer_idx][WEIGHT_ATTENTION], hidden_size, hidden_size);
//			matcopy_row_to_col_float(gpu_context->h_weight[layer_idx][WEIGHT_ATTENTION_BIAS], bert_state->weight[layer_idx][WEIGHT_ATTENTION_BIAS], seq_length, hidden_size);
			matcopy_row_to_col_float(gpu_context->h_weight[layer_idx][WEIGHT_ATTENTION_GAMMA], bert_state->weight[layer_idx][WEIGHT_ATTENTION_GAMMA], seq_length, hidden_size);
			matcopy_row_to_col_float(gpu_context->h_weight[layer_idx][WEIGHT_ATTENTION_BETA], bert_state->weight[layer_idx][WEIGHT_ATTENTION_BETA], seq_length, hidden_size);
//			matcopy_row_to_col_float(gpu_context->h_weight[layer_idx][WEIGHT_PREV_FFW], bert_state->weight[layer_idx][WEIGHT_PREV_FFW], hidden_size, feedforward_size);
//			matcopy_row_to_col_float(gpu_context->h_weight[layer_idx][WEIGHT_PREV_FFB], bert_state->weight[layer_idx][WEIGHT_PREV_FFB], seq_length, feedforward_size);
//			matcopy_row_to_col_float(gpu_context->h_weight[layer_idx][WEIGHT_POST_FFW], bert_state->weight[layer_idx][WEIGHT_POST_FFW], feedforward_size, hidden_size);
//			matcopy_row_to_col_float(gpu_context->h_weight[layer_idx][WEIGHT_POST_FFB], bert_state->weight[layer_idx][WEIGHT_POST_FFB], seq_length, hidden_size);
			matcopy_row_to_col_float(gpu_context->h_weight[layer_idx][WEIGHT_FF_GAMMA], bert_state->weight[layer_idx][WEIGHT_FF_GAMMA], seq_length, hidden_size);
			matcopy_row_to_col_float(gpu_context->h_weight[layer_idx][WEIGHT_FF_BETA], bert_state->weight[layer_idx][WEIGHT_FF_BETA], seq_length, hidden_size);

			for (int head_idx = 0; head_idx < bert_state->num_heads; head_idx++) {
				matcopy_row_to_col_float(&gpu_context->h_weight_attention_fc_splitted[layer_idx][head_idx * head_size * hidden_size],
				                         bert_state->weight_attention_fc_splitted[layer_idx][head_idx], head_size, hidden_size);
			}
			matcopy_row_to_col_float(gpu_context->h_weight_attention_fc_bias_splitted[layer_idx], bert_state->weight_attention_fc_bias_splitted[layer_idx], seq_length, hidden_size);

			for (int ffw_chunk_idx = 0; ffw_chunk_idx < bert_state->num_heads; ffw_chunk_idx++) {
				matcopy_row_to_col_float(&gpu_context->h_weight_ffw_prev[layer_idx][ffw_chunk_idx * hidden_size * ffw_chunk_size],
				                         bert_state->weight_ffw_prev_splitted[layer_idx][ffw_chunk_idx], hidden_size, ffw_chunk_size);
				matcopy_row_to_col_float(&gpu_context->h_weight_ffw_prev_bias[layer_idx][ffw_chunk_idx * seq_length * ffw_chunk_size],
				                         bert_state->weight_ffw_prev_bias_splitted[layer_idx][ffw_chunk_idx], seq_length, ffw_chunk_size);
				matcopy_row_to_col_float(&gpu_context->h_weight_ffw_post_splitted[layer_idx][ffw_chunk_idx * ffw_chunk_size * hidden_size],
				                         bert_state->weight_ffw_post_splitted[layer_idx][ffw_chunk_idx], ffw_chunk_size, hidden_size);
			}
			matcopy_row_to_col_float(gpu_context->h_weight_ffw_post_bias_splitted[layer_idx], bert_state->weight_ffw_post_bias_splitted[layer_idx], seq_length, hidden_size);
		}
	} else if (params->execution_mode == EXEC_MODE_PERF_TEST) {
		for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
			/// We don't need transpose (because we assume data are random values)
			memcpy(gpu_context->h_attention_mask[batch_idx], bert_state->attention_mask[batch_idx], seq_length * seq_length * sizeof(float));
			memcpy(gpu_context->h_input[batch_idx], bert_state->embedding_output[batch_idx], seq_length * hidden_size * sizeof(float));
		}

		for (int layer_idx = 0; layer_idx < num_layer; layer_idx++) {
			memcpy(gpu_context->h_weight[layer_idx][WEIGHT_QLW], bert_state->weight[layer_idx][WEIGHT_QLW], hidden_size * hidden_size * sizeof(float));
			memcpy(gpu_context->h_weight[layer_idx][WEIGHT_QLB], bert_state->weight[layer_idx][WEIGHT_QLB], seq_length * hidden_size * sizeof(float));
			memcpy(gpu_context->h_weight[layer_idx][WEIGHT_KLW], bert_state->weight[layer_idx][WEIGHT_KLW], hidden_size * hidden_size * sizeof(float));
			memcpy(gpu_context->h_weight[layer_idx][WEIGHT_KLB], bert_state->weight[layer_idx][WEIGHT_KLB], seq_length * hidden_size * sizeof(float));
			memcpy(gpu_context->h_weight[layer_idx][WEIGHT_VLW], bert_state->weight[layer_idx][WEIGHT_VLW], hidden_size * hidden_size * sizeof(float));
			memcpy(gpu_context->h_weight[layer_idx][WEIGHT_VLB], bert_state->weight[layer_idx][WEIGHT_VLB], seq_length * hidden_size * sizeof(float));
//			memcpy(gpu_context->h_weight[layer_idx][WEIGHT_ATTENTION], bert_state->weight[layer_idx][WEIGHT_ATTENTION], hidden_size * hidden_size * sizeof(float));
//			memcpy(gpu_context->h_weight[layer_idx][WEIGHT_ATTENTION_BIAS], bert_state->weight[layer_idx][WEIGHT_ATTENTION_BIAS], seq_length * hidden_size * sizeof(float));
			memcpy(gpu_context->h_weight[layer_idx][WEIGHT_ATTENTION_GAMMA], bert_state->weight[layer_idx][WEIGHT_ATTENTION_GAMMA], seq_length * hidden_size * sizeof(float));
			memcpy(gpu_context->h_weight[layer_idx][WEIGHT_ATTENTION_BETA], bert_state->weight[layer_idx][WEIGHT_ATTENTION_BETA], seq_length * hidden_size * sizeof(float));
//			memcpy(gpu_context->h_weight[layer_idx][WEIGHT_PREV_FFW], bert_state->weight[layer_idx][WEIGHT_PREV_FFW], hidden_size * feedforward_size * sizeof(float));
//			memcpy(gpu_context->h_weight[layer_idx][WEIGHT_PREV_FFB], bert_state->weight[layer_idx][WEIGHT_PREV_FFB], seq_length * feedforward_size * sizeof(float));
//			memcpy(gpu_context->h_weight[layer_idx][WEIGHT_POST_FFW], bert_state->weight[layer_idx][WEIGHT_POST_FFW], feedforward_size * hidden_size * sizeof(float));
//			memcpy(gpu_context->h_weight[layer_idx][WEIGHT_POST_FFB], bert_state->weight[layer_idx][WEIGHT_POST_FFB], seq_length * hidden_size * sizeof(float));
			memcpy(gpu_context->h_weight[layer_idx][WEIGHT_FF_GAMMA], bert_state->weight[layer_idx][WEIGHT_FF_GAMMA], seq_length * hidden_size * sizeof(float));
			memcpy(gpu_context->h_weight[layer_idx][WEIGHT_FF_BETA], bert_state->weight[layer_idx][WEIGHT_FF_BETA], seq_length * hidden_size * sizeof(float));

			for (int head_idx = 0; head_idx < bert_state->num_heads; head_idx++) {
				memcpy(&gpu_context->h_weight_attention_fc_splitted[layer_idx][head_idx * head_size * hidden_size],
				       bert_state->weight_attention_fc_splitted[layer_idx][head_idx], head_size * hidden_size * sizeof(float));
			}
			memcpy(gpu_context->h_weight_attention_fc_bias_splitted[layer_idx], bert_state->weight_attention_fc_bias_splitted[layer_idx], seq_length * hidden_size * sizeof(float));
			for (int ffw_chunk_idx = 0; ffw_chunk_idx < bert_state->num_heads; ffw_chunk_idx++) {
				memcpy(&gpu_context->h_weight_ffw_prev[layer_idx][ffw_chunk_idx * hidden_size * ffw_chunk_size],
				       bert_state->weight_ffw_prev_splitted[layer_idx][ffw_chunk_idx], hidden_size * ffw_chunk_size * sizeof(float));
				memcpy(&gpu_context->h_weight_ffw_prev_bias[layer_idx][ffw_chunk_idx * seq_length * ffw_chunk_size],
				       bert_state->weight_ffw_prev_bias_splitted[layer_idx][ffw_chunk_idx], seq_length * ffw_chunk_size * sizeof(float));
				memcpy(&gpu_context->h_weight_ffw_post_splitted[layer_idx][ffw_chunk_idx * ffw_chunk_size * hidden_size],
				       bert_state->weight_ffw_post_splitted[layer_idx][ffw_chunk_idx], ffw_chunk_size * hidden_size * sizeof(float));
			}
			memcpy(gpu_context->h_weight_ffw_post_bias_splitted[layer_idx], bert_state->weight_ffw_post_bias_splitted[layer_idx], seq_length * hidden_size * sizeof(float));
		}
	} else {
		assert(0);
	}

	/// Memcpy (host -> device)
	///  1. input values
	for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
		if ((cuda_rc = cudaMemcpy(gpu_context->d_input[batch_idx], gpu_context->h_input[batch_idx],
				seq_length * hidden_size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy (input) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
	}
	///  2. Attention mask
	for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
		if ((cuda_rc = cudaMemcpy(gpu_context->d_attention_mask[batch_idx], gpu_context->h_attention_mask[batch_idx],
				seq_length * seq_length * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy (d_attention_mask) (cuda_rc: %d)\n", cuda_rc); goto err;
		}
	}

	/// 3. One vector & one matrix
//	for (int i = 0; i < 1 * hidden_size; i ++) { gpu_context->h_onevec[i] = 1.0f; }
//	if ((cuda_rc = cudaMemcpy(gpu_context->d_onevec, gpu_context->h_onevec,
//			1 * hidden_size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy (d_onevec) (cuda_rc: %d)\n", cuda_rc); goto err;
//	}
//	for (int i = 0; i < hidden_size * hidden_size; i ++) { gpu_context->h_onemat[i] = 1.0f; }
//	if ((cuda_rc = cudaMemcpy(gpu_context->d_onemat, gpu_context->h_onemat,
//			hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy (d_onemat) (cuda_rc: %d)\n", cuda_rc); goto err;
//	}
	cudaDeviceSynchronize();

	return 0;
err:
	return -1;
}

static inline int cuda_layer_norm(int batch_idx, int thread_block_size,
		gpu_cuda_context_t *gpu_context, float *in, float *out, int M_dim, int N_dim,
		float *gamma_mat, float *beta_mat) {
	dim3 dimBlock(thread_block_size);
	dim3 dimGrid(M_dim * N_dim / thread_block_size);

	/// Calculate mean
	cudaMemset(gpu_context->d_buf_layernorm_nrm_v[batch_idx], 0, M_dim * 1 * sizeof(float));
	g_layer_mean<<<dimGrid, dimBlock, 0>>>(
			in, gpu_context->d_buf_layernorm_nrm_v[batch_idx],
			M_dim, N_dim);
	g_layer_minus<<<dimGrid, dimBlock, 0>>>(
			in, gpu_context->d_buf_layernorm_nrm_v[batch_idx], in,
			M_dim, N_dim);

	/// Calculate norm2
	cudaMemset(gpu_context->d_buf_layernorm_nrm_v[batch_idx], 0, M_dim * 1 * sizeof(float));
	g_layer_snrm2<<<dimGrid, dimBlock, 0>>>(
			in, gpu_context->d_buf_layernorm_nrm_v[batch_idx],
			M_dim, N_dim);
	g_sqrt<<<dimGrid, dimBlock, 0>>>(
			gpu_context->d_buf_layernorm_nrm_v[batch_idx], M_dim * 1);

	/// var calculation & gamma beta
	g_layer_norm_gamma_beta<<<dimGrid, dimBlock, 0>>>(
			in, out, gpu_context->d_buf_layernorm_nrm_v[batch_idx],
			gamma_mat, beta_mat,
			M_dim, N_dim);

	return 0;
//err:
//	return -1;
}
int cuda_multi_bert_main(multi_gpu_thread_arg_t* multi_gpu_arg) {
	const int gpu_id = multi_gpu_arg->gpu_id;
	Params *params = multi_gpu_arg->params;
	BERT_State *bert_state = multi_gpu_arg->bert_state;
	gpu_cuda_context_t *gpu_context = &multi_gpu_arg->gpu_contexts[gpu_id];
	gpu_cuda_context_t *gpu_context_gpu0 = &multi_gpu_arg->gpu_contexts[0];
	const int num_batch = bert_state->num_batch;
	const int num_layer = bert_state->num_layer;
	const int seq_length = bert_state->seq_length;
	const int hidden_size = bert_state->hidden_size;
	const int head_size = bert_state->hidden_size / bert_state->num_heads;
	const int ffw_chunk_size = bert_state->feedforwardsize / bert_state->num_heads;
	const int hidden_per_gpu = bert_state->hidden_size / multi_gpu_arg->gpu_num;
	const int ffw_per_gpu = bert_state->feedforwardsize / multi_gpu_arg->gpu_num;
	const int num_head_per_gpu = bert_state->num_heads / multi_gpu_arg->gpu_num;
	const int num_ffchunk_per_gpu = bert_state->num_heads / multi_gpu_arg->gpu_num;
	cudaError_t cuda_rc;
	cublasStatus_t cublas_rc;
	const float score_norm_factor = (1.0f / sqrtf((float)head_size));


	/// Assume d_input is already initialized
	for (int layer_idx = 0; layer_idx < num_layer; layer_idx++) {
		/////////////
		/// Q_GEN
		/////////////
		/// Q_GEN memcpy
		if (params->memcpy_mode == MEMCPY_MODE_ALL_OVERHEAD) {
			// copy weight_QLW
			if ((cuda_rc = cudaMemcpy(
					&gpu_context->d_weight[layer_idx][WEIGHT_QLW][gpu_id * hidden_size * hidden_per_gpu],
					&gpu_context->h_weight[layer_idx][WEIGHT_QLW][gpu_id * hidden_size * hidden_per_gpu],
					hidden_size * hidden_per_gpu * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
				fprintf(stderr, "[Q_GEN] <QLW> cudaMemcpy (cuda_rc: %d)\n", cuda_rc);
				goto err;
			}
			// copy weight_QLB
			if ((cuda_rc = cudaMemcpy(
					&gpu_context->d_weight[layer_idx][WEIGHT_QLB][gpu_id * seq_length * hidden_per_gpu],
					&gpu_context->h_weight[layer_idx][WEIGHT_QLB][gpu_id * seq_length * hidden_per_gpu],
					seq_length * hidden_per_gpu * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
				fprintf(stderr, "[Q_GEN] <QLB> cudaMemcpy (cuda_rc: %d)\n", cuda_rc);
				goto err;
			}
		}
		/// Q_GEN matmul & bias
		for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
			float alpha = 1., beta = 0.;
			if ((cublas_rc = cublasSgemm(gpu_context->cublas_handles[0], CUBLAS_OP_N, CUBLAS_OP_N,
					seq_length, hidden_per_gpu, hidden_size,
					&alpha, gpu_context->d_input[batch_idx], seq_length,
					&gpu_context->d_weight[layer_idx][WEIGHT_QLW][gpu_id * hidden_size * hidden_per_gpu], hidden_size,
					&beta, &gpu_context->d_buf_query[batch_idx][gpu_id * seq_length * hidden_per_gpu], seq_length)) != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "[Q_GEN] cublasSgemm (cublas_rc: %d)\n", cublas_rc); goto err;
			}

			if ((cublas_rc = cublasSaxpy(gpu_context->cublas_handles[0], seq_length * hidden_per_gpu, &alpha,
					&gpu_context->d_weight[layer_idx][WEIGHT_QLB][gpu_id * seq_length * hidden_per_gpu], 1,
					&gpu_context->d_buf_query[batch_idx][gpu_id * seq_length * hidden_per_gpu], 1)) != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "[Q_GEN] cublasSaxpy (cublas_rc: %d)\n", cublas_rc); goto err;
			}
		}


		/////////////
		/// K_GEN
		/////////////
		/// K_GEN memcpy
		if (params->memcpy_mode == MEMCPY_MODE_ALL_OVERHEAD) {
			// copy weight_KLW
			if ((cuda_rc = cudaMemcpy(
					&gpu_context->d_weight[layer_idx][WEIGHT_KLW][gpu_id * hidden_size * hidden_per_gpu],
					&gpu_context->h_weight[layer_idx][WEIGHT_KLW][gpu_id * hidden_size * hidden_per_gpu],
					hidden_size * hidden_per_gpu * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
				fprintf(stderr, "[K_GEN] <KLW> cudaMemcpy (cuda_rc: %d)\n", cuda_rc);
				goto err;
			}
			// copy weight_KLB
			if ((cuda_rc = cudaMemcpy(
					&gpu_context->d_weight[layer_idx][WEIGHT_KLB][gpu_id * seq_length * hidden_per_gpu],
					&gpu_context->h_weight[layer_idx][WEIGHT_KLB][gpu_id * seq_length * hidden_per_gpu],
					seq_length * hidden_per_gpu * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
				fprintf(stderr, "[K_GEN] <KLB> cudaMemcpy (cuda_rc: %d)\n", cuda_rc);
				goto err;
			}
		}
		/// K_GEN matmul & bias
		for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
			float alpha = 1., beta = 0.;
			if ((cublas_rc = cublasSgemm(gpu_context->cublas_handles[0], CUBLAS_OP_N, CUBLAS_OP_N,
					seq_length, hidden_per_gpu, hidden_size,
					&alpha, gpu_context->d_input[batch_idx], seq_length,
					&gpu_context->d_weight[layer_idx][WEIGHT_KLW][gpu_id * hidden_size * hidden_per_gpu], hidden_size,
					&beta, &gpu_context->d_buf_key[batch_idx][gpu_id * seq_length * hidden_per_gpu], seq_length)) != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "[K_GEN] cublasSgemm (cublas_rc: %d)\n", cublas_rc); goto err;
			}

			if ((cublas_rc = cublasSaxpy(gpu_context->cublas_handles[0], seq_length * hidden_per_gpu, &alpha,
					&gpu_context->d_weight[layer_idx][WEIGHT_KLB][gpu_id * seq_length * hidden_per_gpu], 1,
					&gpu_context->d_buf_key[batch_idx][gpu_id * seq_length * hidden_per_gpu], 1)) != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "[K_GEN] cublasSaxpy (cublas_rc: %d)\n", cublas_rc); goto err;
			}
		}


		/////////////
		/// Score_calculation
		/////////////
		for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
			float alpha = 1., beta = 0.;
			if ((cublas_rc = cublasSgemmStridedBatched(gpu_context->cublas_handles[0], CUBLAS_OP_N, CUBLAS_OP_T,
					seq_length, seq_length, head_size,
					&alpha, &gpu_context->d_buf_query[batch_idx][seq_length * head_size * gpu_id * num_head_per_gpu],
					seq_length, seq_length * head_size,
					&gpu_context->d_buf_key[batch_idx][seq_length * head_size * gpu_id * num_head_per_gpu],
					seq_length, seq_length * head_size,
					&beta, &gpu_context->d_buf_score[batch_idx][seq_length * seq_length * gpu_id * num_head_per_gpu],
					seq_length, seq_length * seq_length, num_head_per_gpu)) != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "[Score_calculation] cublasSgemmStridedBatched (cublas_rc: %d)\n", cublas_rc); goto err;
			}
		}


		/////////////
		/// socre_norm & ATTENTION_LAYER_MASK_SUB
		/////////////
		for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
			dim3 dimBlock(params->thread_block_size);
			dim3 dimGrid(num_head_per_gpu * seq_length * seq_length / params->thread_block_size);

			g_score_norm_layer_mask<<<dimGrid, dimBlock, 0>>>(
					&gpu_context->d_buf_score[batch_idx][gpu_id * num_head_per_gpu * seq_length * seq_length], score_norm_factor,
					gpu_context->d_attention_mask[batch_idx], seq_length, seq_length, num_head_per_gpu);
		}


		/////////////
		/// Softmax
		/////////////
		for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
			dim3 dimBlock(params->thread_block_size);
			dim3 dimGrid(num_head_per_gpu * seq_length * seq_length / params->thread_block_size);

			/// exp_sum
			cudaMemset(&gpu_context->d_buf_expsum[batch_idx][gpu_id * num_head_per_gpu * seq_length * 1],
					0, num_head_per_gpu * seq_length * 1 * sizeof(float));
			g_exp_sum<<<dimGrid, dimBlock, 0>>>(
					&gpu_context->d_buf_score[batch_idx][gpu_id * num_head_per_gpu * seq_length * seq_length],
					&gpu_context->d_buf_softmax[batch_idx][gpu_id * num_head_per_gpu * seq_length * seq_length],
					&gpu_context->d_buf_expsum[batch_idx][gpu_id * num_head_per_gpu * seq_length * 1],
					seq_length, seq_length, num_head_per_gpu);

			// div
			g_normalize<<<dimGrid, dimBlock, 0>>>(
					&gpu_context->d_buf_softmax[batch_idx][gpu_id * num_head_per_gpu * seq_length * seq_length],
					&gpu_context->d_buf_expsum[batch_idx][gpu_id * num_head_per_gpu * seq_length * 1],
					seq_length, seq_length, num_head_per_gpu);
		}


		/////////////
		/// V_GEN
		/////////////
		/// V_GEN memcpy
		if (params->memcpy_mode == MEMCPY_MODE_ALL_OVERHEAD) {
			// copy weight_VLW
			if ((cuda_rc = cudaMemcpy(
					&gpu_context->d_weight[layer_idx][WEIGHT_VLW][gpu_id * hidden_size * hidden_per_gpu],
					&gpu_context->h_weight[layer_idx][WEIGHT_VLW][gpu_id * hidden_size * hidden_per_gpu],
					hidden_size * hidden_per_gpu * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
				fprintf(stderr, "[V_GEN] <VLW> cudaMemcpy (cuda_rc: %d)\n", cuda_rc);
				goto err;
			}
			// copy weight_VLB
			if ((cuda_rc = cudaMemcpy(
					&gpu_context->d_weight[layer_idx][WEIGHT_VLB][gpu_id * seq_length * hidden_per_gpu],
					&gpu_context->h_weight[layer_idx][WEIGHT_VLB][gpu_id * seq_length * hidden_per_gpu],
					seq_length * hidden_per_gpu * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
				fprintf(stderr, "[V_GEN] <VLB> cudaMemcpy (cuda_rc: %d)\n", cuda_rc);
				goto err;
			}
		}
		/// V_GEN matmul & bias
		for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
			float alpha = 1., beta = 0.;
			if ((cublas_rc = cublasSgemm(gpu_context->cublas_handles[0], CUBLAS_OP_N, CUBLAS_OP_N,
					seq_length, hidden_per_gpu, hidden_size,
					&alpha, gpu_context->d_input[batch_idx], seq_length,
					&gpu_context->d_weight[layer_idx][WEIGHT_VLW][gpu_id * hidden_size * hidden_per_gpu], hidden_size,
					&beta, &gpu_context->d_buf_value[batch_idx][gpu_id * seq_length * hidden_per_gpu], seq_length)) != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "[V_GEN] cublasSgemm (cublas_rc: %d)\n", cublas_rc); goto err;
			}

			if ((cublas_rc = cublasSaxpy(gpu_context->cublas_handles[0], seq_length * hidden_per_gpu, &alpha,
					&gpu_context->d_weight[layer_idx][WEIGHT_VLB][gpu_id * seq_length * hidden_per_gpu], 1,
					&gpu_context->d_buf_value[batch_idx][gpu_id * seq_length * hidden_per_gpu], 1)) != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "[V_GEN] cublasSaxpy (cublas_rc: %d)\n", cublas_rc); goto err;
			}
		}


		/////////////
		/// Weighted_Sum
		/////////////
		for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
			float alpha = 1., beta = 0.;
			if ((cublas_rc = cublasSgemmStridedBatched(gpu_context->cublas_handles[0], CUBLAS_OP_N, CUBLAS_OP_N,
					seq_length, head_size, seq_length,
					&alpha, &gpu_context->d_buf_softmax[batch_idx][seq_length * seq_length * gpu_id * num_head_per_gpu],
					seq_length, seq_length * seq_length,
					&gpu_context->d_buf_value[batch_idx][seq_length * head_size * gpu_id * num_head_per_gpu],
					seq_length, seq_length * head_size,
					&beta, &gpu_context->d_buf_att[batch_idx][seq_length * head_size * gpu_id * num_head_per_gpu],
					seq_length, seq_length * head_size, num_head_per_gpu)) != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "[Weighted_Sum] cublasSgemmStridedBatched (cublas_rc: %d)\n", cublas_rc); goto err;
			}
		}


		/////////////
		/// ATTENTION_FC
		/////////////
		/// ATTENTION_FC memcpy
		if (params->memcpy_mode == MEMCPY_MODE_ALL_OVERHEAD) {
			// copy WEIGHT_ATTENTION
			if ((cuda_rc = cudaMemcpy(
					&gpu_context->d_weight_attention_fc_splitted[layer_idx][gpu_id * hidden_per_gpu * hidden_size],
					&gpu_context->h_weight_attention_fc_splitted[layer_idx][gpu_id * hidden_per_gpu * hidden_size],
					hidden_per_gpu * hidden_size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
				fprintf(stderr, "[ATTENTION_FC] <WEIGHT_ATTENTION> cudaMemcpy (cuda_rc: %d)\n", cuda_rc);
				goto err;
			}
			if (gpu_id == 0) {
				// copy WEIGHT_ATTENTION_BIAS
				// cooperate between streams to load the bias value (since we use this bias values at reduce_sum only)
				if ((cuda_rc = cudaMemcpy(
						&gpu_context->d_weight_attention_fc_bias_splitted[layer_idx][gpu_id * seq_length * hidden_size],
						&gpu_context->h_weight_attention_fc_bias_splitted[layer_idx][gpu_id * seq_length * hidden_size],
						seq_length * hidden_size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
					fprintf(stderr, "[Feedforward_POST] <weight_FFW_POST_BIAS> cudaMemcpy (cuda_rc: %d)\n", cuda_rc);
					goto err;
				}
			}
		}

		/// ATTENTION_FC matmul & bias
		for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
			float alpha = 1., beta = 0.;
			if ((cublas_rc = cublasSgemmStridedBatched(gpu_context->cublas_handles[0], CUBLAS_OP_N, CUBLAS_OP_N,
					seq_length, hidden_size, head_size,
					&alpha, &gpu_context->d_buf_att[batch_idx][seq_length * head_size * gpu_id * num_head_per_gpu],
					seq_length, seq_length * head_size,
					&gpu_context->d_weight_attention_fc_splitted[layer_idx][head_size * hidden_size * gpu_id * num_head_per_gpu],
					head_size, head_size * hidden_size,
					&beta, &gpu_context->d_buf_att_fc_result_split[batch_idx][seq_length * hidden_size * gpu_id * num_head_per_gpu],
					seq_length, seq_length * hidden_size, num_head_per_gpu)) != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "[ATTENTION_FC] cublasSgemmStridedBatched (cublas_rc: %d)\n", cublas_rc); goto err;
			}
		}


		/////////////
		/// ATTENTION_FC_Partial_Reduce_Sum (Since we partition it)
		/////////////
		for (int head_idx = (num_head_per_gpu * gpu_id + 1); head_idx < num_head_per_gpu * (gpu_id + 1); head_idx++) {
			for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
				float alpha = 1.;
				if ((cublas_rc = cublasSaxpy(gpu_context->cublas_handles[0], seq_length * hidden_size, &alpha,
						&gpu_context->d_buf_att_fc_result_split[batch_idx][head_idx * seq_length * hidden_size], 1,
						&gpu_context->d_buf_att_fc_result_split[batch_idx][num_head_per_gpu * gpu_id * seq_length * hidden_size], 1)) != CUBLAS_STATUS_SUCCESS) {
						fprintf(stderr, "[ATTENTION_FC_Reduce_Sum] cublasSaxpy (cublas_rc: %d)\n", cublas_rc); goto err;
				}
			}
		}


		///////////////////////////////////////////////
		/// Send partial results to GPU 0 (cudaMemcpyPeer)
		if (gpu_id != 0) {
			if (params->memcpy_mode != MEMCPY_MODE_NO_ALL_OVERHEAD) {
				for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
					if ((cuda_rc = cudaMemcpy(
							&gpu_context_gpu0->d_buf_att_fc_result_split[batch_idx][num_head_per_gpu * gpu_id * seq_length * hidden_size],
							&gpu_context->d_buf_att_fc_result_split[batch_idx][num_head_per_gpu * gpu_id * seq_length * hidden_size],
							seq_length * hidden_size * sizeof(float), cudaMemcpyDeviceToDevice)) != cudaSuccess) {
						fprintf(stderr, "[Send partial results] d_buf_att_fc_result_split cudaMemcpyPeer (cuda_rc: %d)\n", cuda_rc);
						goto err;
					}
//					if ((cuda_rc = cudaMemcpyPeer(
//							&gpu_context_gpu0->d_buf_att_fc_result_split[batch_idx][num_head_per_gpu * gpu_id * seq_length * hidden_size], 0,
//							&gpu_context->d_buf_att_fc_result_split[batch_idx][num_head_per_gpu * gpu_id * seq_length * hidden_size], gpu_id,
//							seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
//						fprintf(stderr, "[Send partial results] d_buf_att_fc_result_split cudaMemcpyPeer (cuda_rc: %d)\n", cuda_rc);
//						goto err;
//					}
				}
			}
		}


		pthread_barrier_wait(multi_gpu_arg->multi_gpu_barrier_local_att_fc_rsum);
		if (gpu_id == 0) {
			/////////////
			/// ATTENTION_FC_Reduce_Sum (Since we partition it)
			/////////////
			for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
				float alpha = 1.;
				for (int t_gpu_idx = 1; t_gpu_idx < params->num_gpus; t_gpu_idx++) {
					if ((cublas_rc = cublasSaxpy(gpu_context->cublas_handles[0], seq_length * hidden_size, &alpha,
							&gpu_context->d_buf_att_fc_result_split[batch_idx][num_head_per_gpu * t_gpu_idx * seq_length * hidden_size], 1,
							gpu_context->d_buf_att_fc_result_split[batch_idx], 1)) != CUBLAS_STATUS_SUCCESS) {
						fprintf(stderr, "[ATTENTION_FC_Reduce_Sum] cublasSaxpy (cublas_rc: %d)\n", cublas_rc); goto err;
					}
				}

				if ((cublas_rc = cublasSaxpy(gpu_context->cublas_handles[0], seq_length * hidden_size, &alpha,
						gpu_context->d_weight_attention_fc_bias_splitted[layer_idx], 1,
						gpu_context->d_buf_att_fc_result_split[batch_idx], 1)) != CUBLAS_STATUS_SUCCESS) {
					fprintf(stderr, "[ATTENTION_FC_Reduce_Sum] BIAS cublasSaxpy (cublas_rc: %d)\n", cublas_rc); goto err;
				}
			}


			/////////////
			/// ATTENTION_RESIDUAL
			/////////////
			for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
				float alpha = 1.;
				if ((cublas_rc = cublasSaxpy(gpu_context->cublas_handles[0], seq_length * hidden_size, &alpha,
						gpu_context->d_input[batch_idx], 1,
						gpu_context->d_buf_att_fc_result_split[batch_idx], 1)) != CUBLAS_STATUS_SUCCESS) {
					fprintf(stderr, "[ATTENTION_RESIDUAL] cublasSaxpy (cublas_rc: %d)\n", cublas_rc); goto err;
				}
			}


			/////////////
			/// ATTENTION_NORM
			/////////////
			if (params->memcpy_mode == MEMCPY_MODE_ALL_OVERHEAD) {
				// copy WEIGHT_ATTENTION_GAMMA
				if ((cuda_rc = cudaMemcpy(gpu_context->d_weight[layer_idx][WEIGHT_ATTENTION_GAMMA],
						gpu_context->h_weight[layer_idx][WEIGHT_ATTENTION_GAMMA],
						seq_length * hidden_size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
					fprintf(stderr, "[ATTENTION_NORM] <WEIGHT_ATTENTION_GAMMA> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
				}
				// copy WEIGHT_ATTENTION_BETA
				if ((cuda_rc = cudaMemcpy(gpu_context->d_weight[layer_idx][WEIGHT_ATTENTION_BETA],
						gpu_context->h_weight[layer_idx][WEIGHT_ATTENTION_BETA],
						seq_length * hidden_size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
					fprintf(stderr, "[ATTENTION_NORM] <WEIGHT_ATTENTION_BETA> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
				}
			}
			/// Do calculation
			for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
				cuda_layer_norm(batch_idx, params->thread_block_size,
								gpu_context, gpu_context->d_buf_att_fc_result_split[batch_idx], gpu_context->d_buf_att_layernorm[batch_idx],
								seq_length, hidden_size,
								gpu_context->d_weight[layer_idx][WEIGHT_ATTENTION_GAMMA], gpu_context->d_weight[layer_idx][WEIGHT_ATTENTION_BETA]);
			}
		}


		/// Broadcast output to GPUs (cudaMemcpyPeer)
		pthread_barrier_wait(multi_gpu_arg->multi_gpu_barrier_local_att_fc_rsum_rescopy);
		if (gpu_id != 0) {
			if (params->memcpy_mode != MEMCPY_MODE_NO_ALL_OVERHEAD) {
				for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
					if ((cuda_rc = cudaMemcpy(
							&gpu_context->d_buf_att_layernorm[batch_idx][0],
							&gpu_context_gpu0->d_buf_att_layernorm[batch_idx][0],
							seq_length * hidden_size * sizeof(float), cudaMemcpyDeviceToDevice)) != cudaSuccess) {
						fprintf(stderr, "[copy_res] <d_buf_att_layernorm> cudaMemcpyPeer (cuda_rc: %d)\n", cuda_rc);
						goto err;
					}
//					if ((cuda_rc = cudaMemcpyPeer(
//							&gpu_context->d_buf_att_layernorm[batch_idx][0], gpu_id,
//							&gpu_context_gpu0->d_buf_att_layernorm[batch_idx][0], 0,
//							seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
//						fprintf(stderr, "[copy_res] <d_buf_att_layernorm> cudaMemcpyPeer (cuda_rc: %d)\n", cuda_rc);
//						goto err;
//					}
				}
			}
		}
		pthread_barrier_wait(multi_gpu_arg->multi_gpu_barrier_local_att_fc_rsum_rescopy_done);
		///////////////////////////////////////////////


		/////////////
		/// Feedforward_PREV
		/////////////
		/// Feedforward_PREV memcpy
		if (params->memcpy_mode == MEMCPY_MODE_ALL_OVERHEAD) {
			// copy weight_FFW_PREV
			if ((cuda_rc = cudaMemcpy(&gpu_context->d_weight_ffw_prev[layer_idx][gpu_id * hidden_size * ffw_per_gpu],
			                          &gpu_context->h_weight_ffw_prev[layer_idx][gpu_id * hidden_size * ffw_per_gpu],
			                          hidden_size * ffw_per_gpu * sizeof(float), cudaMemcpyHostToDevice)) !=
			    cudaSuccess) {
				fprintf(stderr, "[Feedforward_PREV] <weight_FFW_PREV> cudaMemcpy (cuda_rc: %d)\n", cuda_rc);
				goto err;
			}
			// copy weight_FFW_PREV_BIAS
			if ((cuda_rc = cudaMemcpy(
					&gpu_context->d_weight_ffw_prev_bias[layer_idx][gpu_id * seq_length * ffw_per_gpu],
					&gpu_context->h_weight_ffw_prev_bias[layer_idx][gpu_id * seq_length * ffw_per_gpu],
					seq_length * ffw_per_gpu * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
				fprintf(stderr, "[Feedforward_PREV] <weight_FFW_PREV_BIAS> cudaMemcpy (cuda_rc: %d)\n", cuda_rc);
				goto err;
			}
		}
		/// Feedforward_PREV matmul & bias
		for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
			float alpha = 1., beta = 0.;
			if ((cublas_rc = cublasSgemm(gpu_context->cublas_handles[0], CUBLAS_OP_N, CUBLAS_OP_N,
					seq_length, ffw_per_gpu, hidden_size,
					&alpha, gpu_context->d_buf_att_layernorm[batch_idx], seq_length,
					&gpu_context->d_weight_ffw_prev[layer_idx][gpu_id * hidden_size * ffw_per_gpu], hidden_size,
					&beta, &gpu_context->d_buf_ffw_intermediate[batch_idx][gpu_id * seq_length * ffw_per_gpu], seq_length)) != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "[Feedforward_PREV] cublasSgemm (cublas_rc: %d)\n", cublas_rc); goto err;
			}
			if ((cublas_rc = cublasSaxpy(gpu_context->cublas_handles[0], seq_length * ffw_per_gpu, &alpha,
					&gpu_context->d_weight_ffw_prev_bias[layer_idx][gpu_id * seq_length * ffw_per_gpu], 1,
					&gpu_context->d_buf_ffw_intermediate[batch_idx][gpu_id * seq_length * ffw_per_gpu], 1)) != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "[Feedforward_PREV] cublasSaxpy (cublas_rc: %d)\n", cublas_rc); goto err;
			}
		}


		/////////////
		/// Feedforward_GELU
		/////////////
		for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
			dim3 dimBlock(params->thread_block_size);
			dim3 dimGrid(seq_length * ffw_per_gpu / params->thread_block_size);

			// exp
			g_gelu<<<dimGrid, dimBlock, 0>>>(
					&gpu_context->d_buf_ffw_intermediate[batch_idx][gpu_id * seq_length * ffw_per_gpu],
					&gpu_context->d_buf_ffw_gelu[batch_idx][gpu_id * seq_length * ffw_per_gpu],
					seq_length * ffw_per_gpu);
		}


		/////////////
		/// Feedforward_POST
		/////////////
		/// Feedforward_POST memcpy
		if (params->memcpy_mode == MEMCPY_MODE_ALL_OVERHEAD) {
			// copy weight_FFW_POST
			if ((cuda_rc = cudaMemcpy(
					&gpu_context->d_weight_ffw_post_splitted[layer_idx][gpu_id * ffw_per_gpu * hidden_size],
					&gpu_context->h_weight_ffw_post_splitted[layer_idx][gpu_id * ffw_per_gpu * hidden_size],
					ffw_per_gpu * hidden_size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
				fprintf(stderr, "[Feedforward_POST] <weight_FFW_POST> cudaMemcpy (cuda_rc: %d)\n", cuda_rc);
				goto err;
			}
			if (gpu_id == 0) {
				// copy weight_FFW_POST_BIAS
				// cooperate between streams to load the bias value (since we use this bias values at reduce_sum only)
				if ((cuda_rc = cudaMemcpy(
						&gpu_context->d_weight_ffw_post_bias_splitted[layer_idx][gpu_id * seq_length * hidden_size],
						&gpu_context->h_weight_ffw_post_bias_splitted[layer_idx][gpu_id * seq_length * hidden_size],
						seq_length * hidden_size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
					fprintf(stderr, "[Feedforward_POST] <weight_FFW_POST_BIAS> cudaMemcpy (cuda_rc: %d)\n", cuda_rc);
					goto err;
				}
			}
		}
		/// Feedforward_POST matmul & bias
		for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
			float alpha = 1., beta = 0.;
			if ((cublas_rc = cublasSgemmStridedBatched(gpu_context->cublas_handles[0], CUBLAS_OP_N, CUBLAS_OP_N,
					seq_length, hidden_size, ffw_chunk_size,
					&alpha, &gpu_context->d_buf_ffw_gelu[batch_idx][gpu_id * seq_length * ffw_per_gpu],
					seq_length, seq_length * ffw_chunk_size,
					&gpu_context->d_weight_ffw_post_splitted[layer_idx][gpu_id * ffw_per_gpu * hidden_size],
					ffw_chunk_size, ffw_chunk_size * hidden_size,
					&beta, &gpu_context->d_buf_ffw_result_split[batch_idx][gpu_id * num_ffchunk_per_gpu * seq_length * hidden_size],
					seq_length, seq_length * hidden_size, num_ffchunk_per_gpu)) != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "[Feedforward_POST] cublasSgemmStridedBatched (cublas_rc: %d)\n", cublas_rc); goto err;
			}
		}

		/////////////
		/// Feedforward_Partial_Reduce_Sum (Since we partition it)
		/////////////
		for (int ffw_chunk_idx = (num_ffchunk_per_gpu * gpu_id + 1); ffw_chunk_idx < num_ffchunk_per_gpu * (gpu_id + 1); ffw_chunk_idx++) {
			for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
				float alpha = 1.;
				if ((cublas_rc = cublasSaxpy(gpu_context->cublas_handles[0], seq_length * hidden_size, &alpha,
						&gpu_context->d_buf_ffw_result_split[batch_idx][ffw_chunk_idx * seq_length * hidden_size], 1,
						&gpu_context->d_buf_ffw_result_split[batch_idx][num_ffchunk_per_gpu * gpu_id * seq_length * hidden_size], 1)) != CUBLAS_STATUS_SUCCESS) {
					fprintf(stderr, "[Feedforward_Partial_Reduce_Sum] cublasSaxpy (cublas_rc: %d)\n", cublas_rc); goto err;
				}
			}
		}

		///////////////////////////////////////////////
		/// Send partial results to GPU 0 (cudaMemcpyPeer)
		if (gpu_id != 0) {
			if (params->memcpy_mode != MEMCPY_MODE_NO_ALL_OVERHEAD) {
				for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
					if ((cuda_rc = cudaMemcpy(
							&gpu_context_gpu0->d_buf_ffw_result_split[batch_idx][num_ffchunk_per_gpu * gpu_id * seq_length * hidden_size],
							&gpu_context->d_buf_ffw_result_split[batch_idx][num_ffchunk_per_gpu * gpu_id * seq_length * hidden_size],
							seq_length * hidden_size * sizeof(float), cudaMemcpyDeviceToDevice)) != cudaSuccess) {
						fprintf(stderr, "[Send partial results] d_buf_ffw_result_split cudaMemcpyPeer (cuda_rc: %d)\n", cuda_rc);
						goto err;
					}
//					if ((cuda_rc = cudaMemcpyPeer(
//							&gpu_context_gpu0->d_buf_ffw_result_split[batch_idx][num_ffchunk_per_gpu * gpu_id * seq_length * hidden_size], 0,
//							&gpu_context->d_buf_ffw_result_split[batch_idx][num_ffchunk_per_gpu * gpu_id * seq_length * hidden_size], gpu_id,
//							seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
//						fprintf(stderr, "[Send partial results] d_buf_ffw_result_split cudaMemcpyPeer (cuda_rc: %d)\n", cuda_rc);
//						goto err;
//					}
				}
			}
		}


		pthread_barrier_wait(multi_gpu_arg->multi_gpu_barrier_local_ffw_rsum);
		if (gpu_id == 0) {
			/////////////
			/// Feedforward_Reduce_Sum (Since we partition it)
			/////////////
			for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
				float alpha = 1.;
				for (int t_gpu_idx = 1; t_gpu_idx < params->num_gpus; t_gpu_idx++) {
					if ((cublas_rc = cublasSaxpy(gpu_context->cublas_handles[0], seq_length * hidden_size, &alpha,
							&gpu_context->d_buf_ffw_result_split[batch_idx][num_ffchunk_per_gpu * t_gpu_idx * seq_length * hidden_size], 1,
							gpu_context->d_buf_ffw_result_split[batch_idx], 1)) != CUBLAS_STATUS_SUCCESS) {
						fprintf(stderr, "[Feedforward_Reduce_Sum] cublasSaxpy (cublas_rc: %d)\n", cublas_rc); goto err;
					}
				}
				if ((cublas_rc = cublasSaxpy(gpu_context->cublas_handles[0], seq_length * hidden_size, &alpha,
						gpu_context->d_weight_ffw_post_bias_splitted[layer_idx], 1,
						gpu_context->d_buf_ffw_result_split[batch_idx], 1)) != CUBLAS_STATUS_SUCCESS) {
					fprintf(stderr, "[Feedforward_Reduce_Sum] BIAS cublasSaxpy (cublas_rc: %d)\n", cublas_rc); goto err;
				}
			}


			/////////////
			/// Feedforward_RESIDUAL
			/////////////
			for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
				float alpha = 1.;
				if ((cublas_rc = cublasSaxpy(gpu_context->cublas_handles[0], seq_length * hidden_size, &alpha,
						gpu_context->d_buf_att_layernorm[batch_idx], 1,
						gpu_context->d_buf_ffw_result_split[batch_idx], 1)) != CUBLAS_STATUS_SUCCESS) {
					fprintf(stderr, "[ATTENTION_RESIDUAL] cublasSaxpy (cublas_rc: %d)\n", cublas_rc); goto err;
				}
			}


			/////////////
			/// Feedforward_NORM
			/////////////
			if (params->memcpy_mode == MEMCPY_MODE_ALL_OVERHEAD) {
				// copy WEIGHT_FF_GAMMA
				if ((cuda_rc = cudaMemcpy(gpu_context->d_weight[layer_idx][WEIGHT_FF_GAMMA],
						gpu_context->h_weight[layer_idx][WEIGHT_FF_GAMMA],
						seq_length * hidden_size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
					fprintf(stderr, "[Feedforward_NORM] <WEIGHT_FF_GAMMA> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
				}
				// copy WEIGHT_FF_BETA
				if ((cuda_rc = cudaMemcpy(gpu_context->d_weight[layer_idx][WEIGHT_FF_BETA],
						gpu_context->h_weight[layer_idx][WEIGHT_FF_BETA],
						seq_length * hidden_size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
					fprintf(stderr, "[Feedforward_NORM] <WEIGHT_FF_BETA> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
				}
			}
			/// Do calculation
			for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
				cuda_layer_norm(batch_idx, params->thread_block_size,
								gpu_context, gpu_context->d_buf_ffw_result_split[batch_idx], gpu_context->d_buf_ffw_layernorm[batch_idx],
								seq_length, hidden_size,
								gpu_context->d_weight[layer_idx][WEIGHT_FF_GAMMA], gpu_context->d_weight[layer_idx][WEIGHT_FF_BETA]);
			}
		}


		/// Broadcast output to GPUs (cudaMemcpyPeer)
		pthread_barrier_wait(multi_gpu_arg->multi_gpu_barrier_local_ffw_rsum_rescopy);
		if (params->memcpy_mode != MEMCPY_MODE_NO_ALL_OVERHEAD) {
			for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
				if ((cuda_rc = cudaMemcpy(gpu_context->d_input[batch_idx],
										  gpu_context_gpu0->d_buf_ffw_layernorm[batch_idx],
										  seq_length * hidden_size * sizeof(float), cudaMemcpyDeviceToDevice)) != cudaSuccess) {
					fprintf(stderr, "[copy_res] <d_buf_att_layernorm> cudaMemcpyPeer (cuda_rc: %d)\n", cuda_rc);
					goto err;
				}
//				if ((cuda_rc = cudaMemcpyPeer(gpu_context->d_input[batch_idx], gpu_id,
//											  gpu_context_gpu0->d_buf_ffw_layernorm[batch_idx], 0,
//											  seq_length * hidden_size * sizeof(float))) != cudaSuccess) {
//					fprintf(stderr, "[copy_res] <d_buf_att_layernorm> cudaMemcpyPeer (cuda_rc: %d)\n", cuda_rc);
//					goto err;
//				}
			}
		}
		pthread_barrier_wait(multi_gpu_arg->multi_gpu_barrier_local_ffw_rsum_rescopy_done);
		///////////////////////////////////////////////
	}

	return 0;
err:
	return -1;
}

void dump_gpu_matrix(float *gpu_mem, int M, int N, int gpu_id, const char *prefix_msg) {
	cudaError_t cuda_rc;
	float *host_buf;
	char str_buf[4096];

	if ((cuda_rc = cudaMallocHost((void**)&host_buf, M * N * sizeof(float))) != cudaSuccess) {
		fprintf(stderr, "<dump_gpu_matrix> Fail to cudaMallocHost (reason: %s)\n", cudaGetErrorString(cuda_rc));
		assert(0);
	}

	if ((cuda_rc = cudaMemcpy(host_buf, gpu_mem, M * N * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
		fprintf(stderr, "<dump_gpu_matrix> cudaMemcpy (cuda_rc: %d)\n", cuda_rc);\
		assert(0);
	}

	snprintf(str_buf, sizeof(str_buf), "<%s> [gpu_id: %d] (M: %d, N: %d)\n"
									   "    %+-4.6f %+-4.6f ... %+-4.6f %+-4.6f\n"
									   "    %+-4.6f %+-4.6f ... %+-4.6f %+-4.6f\n"
									   "    ...\n"
									   "    %+-4.6f %+-4.6f ... %+-4.6f %+-4.6f\n"
									   "    %+-4.6f %+-4.6f ... %+-4.6f %+-4.6f\n",
									   prefix_msg, gpu_id, M, N,
									   host_buf[0],   host_buf[M],	   host_buf[(N-2)*M],     host_buf[(N-1)*M],
									   host_buf[1],   host_buf[M+1],   host_buf[(N-2)*M+1],   host_buf[(N-1)*M+1],
									   host_buf[M-2], host_buf[M+M-2], host_buf[(N-2)*M+M-2], host_buf[(N-1)*M+M-2],
									   host_buf[M-1], host_buf[M+M-1], host_buf[(N-2)*M+M-1], host_buf[(N-1)*M+M-1]);
	printf("%s", str_buf);
	cudaFree(host_buf);
}

void sync_all_buf_to_host(BERT_State *bert_state, gpu_cuda_context_t *gpu_context) {
	const int num_batch = bert_state->num_batch;
	const int num_heads = bert_state->num_heads;
	const int seq_length = bert_state->seq_length;
	const int hidden_size = bert_state->hidden_size;
	const int feedforward_size = bert_state->feedforwardsize;
//	const int head_size = bert_state->hidden_size / bert_state->num_heads;
//	const int ffw_chunk_size = bert_state->feedforwardsize / bert_state->num_heads;
	cudaError_t cuda_rc;

	cudaDeviceSynchronize();
	for (int batch_idx = 0; batch_idx < num_batch; batch_idx++) {
		// copy d_input
		if ((cuda_rc = cudaMemcpy(gpu_context->h_input[batch_idx], gpu_context->d_input[batch_idx],
				seq_length * hidden_size * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			fprintf(stderr, "[SYNC_DEBUG] <d_input> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		// copy d_buf_query
		if ((cuda_rc = cudaMemcpy(gpu_context->h_buf_query[batch_idx], gpu_context->d_buf_query[batch_idx],
		                          seq_length * hidden_size * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			fprintf(stderr, "[SYNC_DEBUG] <d_buf_query> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		// copy d_buf_key
		if ((cuda_rc = cudaMemcpy(gpu_context->h_buf_key[batch_idx], gpu_context->d_buf_key[batch_idx],
		                          seq_length * hidden_size * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			fprintf(stderr, "[SYNC_DEBUG] <d_buf_key> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		// copy d_buf_value
		if ((cuda_rc = cudaMemcpy(gpu_context->h_buf_value[batch_idx], gpu_context->d_buf_value[batch_idx],
		                          seq_length * hidden_size * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			fprintf(stderr, "[SYNC_DEBUG] <d_buf_value> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		// copy d_buf_score
		if ((cuda_rc = cudaMemcpy(gpu_context->h_buf_score[batch_idx], gpu_context->d_buf_score[batch_idx],
		                          num_heads * seq_length * seq_length * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			fprintf(stderr, "[SYNC_DEBUG] <d_buf_score> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		// copy d_buf_expsum
		if ((cuda_rc = cudaMemcpy(gpu_context->h_buf_expsum[batch_idx], gpu_context->d_buf_expsum[batch_idx],
		                          num_heads * seq_length * 1 * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			fprintf(stderr, "[SYNC_DEBUG] <d_buf_expsum> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		// copy d_buf_softmax
		if ((cuda_rc = cudaMemcpy(gpu_context->h_buf_softmax[batch_idx], gpu_context->d_buf_softmax[batch_idx],
		                          num_heads * seq_length * seq_length * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			fprintf(stderr, "[SYNC_DEBUG] <d_buf_softmax> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		// copy d_buf_att
		if ((cuda_rc = cudaMemcpy(gpu_context->h_buf_att[batch_idx], gpu_context->d_buf_att[batch_idx],
				seq_length * hidden_size * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			fprintf(stderr, "[SYNC_DEBUG] <d_buf_att> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		// copy d_buf_att_fc_result_split
		if ((cuda_rc = cudaMemcpy(gpu_context->h_buf_att_fc_result_split[batch_idx], gpu_context->d_buf_att_fc_result_split[batch_idx],
		                          num_heads * seq_length * hidden_size * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			fprintf(stderr, "[SYNC_DEBUG] <d_buf_att_fc_result_split> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		// copy d_buf_att_layernorm
		if ((cuda_rc = cudaMemcpy(gpu_context->h_buf_att_layernorm[batch_idx], gpu_context->d_buf_att_layernorm[batch_idx],
				seq_length * hidden_size * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			fprintf(stderr, "[SYNC_DEBUG] <d_buf_att_layernorm> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
		}

		// copy d_buf_ffw_intermediate
		if ((cuda_rc = cudaMemcpy(gpu_context->h_buf_ffw_intermediate[batch_idx], gpu_context->d_buf_ffw_intermediate[batch_idx],
		                          seq_length * feedforward_size * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			fprintf(stderr, "[SYNC_DEBUG] <d_buf_ffw_intermediate> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		// copy d_buf_ffw_gelu
		if ((cuda_rc = cudaMemcpy(gpu_context->h_buf_ffw_gelu[batch_idx], gpu_context->d_buf_ffw_gelu[batch_idx],
		                          seq_length * feedforward_size * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			fprintf(stderr, "[SYNC_DEBUG] <d_buf_ffw_gelu> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		// copy d_buf_ffw_result_split
		if ((cuda_rc = cudaMemcpy(gpu_context->h_buf_ffw_result_split[batch_idx], gpu_context->d_buf_ffw_result_split[batch_idx],
		                          num_heads * seq_length * hidden_size * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			fprintf(stderr, "[SYNC_DEBUG] <d_buf_ffw_result_split> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
		}
		// copy d_buf_ffw_layernorm
		if ((cuda_rc = cudaMemcpy(gpu_context->h_buf_ffw_layernorm[batch_idx], gpu_context->d_buf_ffw_layernorm[batch_idx],
		                          seq_length * hidden_size * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			fprintf(stderr, "[SYNC_DEBUG] <d_buf_ffw_layernorm> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
		}
//		// copy d_buf_layernorm_mean
//		if ((cuda_rc = cudaMemcpy(gpu_context->h_buf_layernorm_mean[batch_idx], gpu_context->d_buf_layernorm_mean[batch_idx],
//				seq_length * hidden_size * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
//			fprintf(stderr, "[SYNC_DEBUG] <d_buf_layernorm_mean> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
//		}
//		// copy d_buf_layernorm_tmp
//		if ((cuda_rc = cudaMemcpy(gpu_context->h_buf_layernorm_tmp[batch_idx], gpu_context->d_buf_layernorm_tmp[batch_idx],
//		                          seq_length * hidden_size * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
//			fprintf(stderr, "[SYNC_DEBUG] <d_buf_layernorm_tmp> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
//		}
		// copy d_buf_layernorm_nrm_v
		if ((cuda_rc = cudaMemcpy(gpu_context->h_buf_layernorm_nrm_v[batch_idx], gpu_context->d_buf_layernorm_nrm_v[batch_idx],
				seq_length * 1 * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			fprintf(stderr, "[SYNC_DEBUG] <d_buf_layernorm_nrm_v> cudaMemcpy (cuda_rc: %d)\n", cuda_rc); goto err;
		}
	}
	cudaDeviceSynchronize();
	printf("[SYNC_DEBUG] buffer sync done\n");

	return;
err:
	fprintf(stderr, "Fail to cudaMemcpy (reason: %s)\n", cudaGetErrorString(cuda_rc));
	assert(0);
}


void cuda_multi_host_context_deinit(BERT_State *bert_state, gpu_cuda_context_t *gpu_context) {
	/// Host memory
//	cudaFreeHost(gpu_context->h_onevec); gpu_context->h_onevec = nullptr;
//	cudaFreeHost(gpu_context->h_onemat); gpu_context->h_onemat = nullptr;

	if (gpu_context->h_attention_mask) {
		for (int batch_idx = 0; batch_idx < bert_state->num_batch; batch_idx++) {
			if (gpu_context->h_attention_mask[batch_idx]) cudaFreeHost(gpu_context->h_attention_mask[batch_idx]);
		}
		free(gpu_context->h_attention_mask); gpu_context->h_attention_mask = nullptr;
	}

	if (gpu_context->h_input) {
		for (int batch_idx = 0; batch_idx < bert_state->num_batch; batch_idx++) {
			if (gpu_context->h_input[batch_idx]) cudaFreeHost(gpu_context->h_input[batch_idx]);
		}
		free(gpu_context->h_input); gpu_context->h_input = nullptr;
	}

	if (gpu_context->h_weight) {
		for (int layer_idx = 0; layer_idx < bert_state->num_layer; layer_idx++) {
			for (int weight_idx = 0; weight_idx < WEIGHT_MAX_NUM; weight_idx++) {
				if (gpu_context->h_weight[layer_idx][weight_idx]) cudaFreeHost(gpu_context->h_weight[layer_idx][weight_idx]);
			}
			free(gpu_context->h_weight[layer_idx]);

			cudaFreeHost(gpu_context->h_weight_attention_fc_splitted[layer_idx]);
			cudaFreeHost(gpu_context->h_weight_attention_fc_bias_splitted[layer_idx]);

			cudaFreeHost(gpu_context->h_weight_ffw_prev[layer_idx]);
			cudaFreeHost(gpu_context->h_weight_ffw_prev_bias[layer_idx]);
			cudaFreeHost(gpu_context->h_weight_ffw_post_splitted[layer_idx]);
			cudaFreeHost(gpu_context->h_weight_ffw_post_bias_splitted[layer_idx]);
		}
		free(gpu_context->h_weight); gpu_context->h_weight = nullptr;
		free(gpu_context->h_weight_attention_fc_splitted); gpu_context->h_weight_attention_fc_splitted = nullptr;
		free(gpu_context->h_weight_attention_fc_bias_splitted); gpu_context->h_weight_attention_fc_bias_splitted = nullptr;
		free(gpu_context->h_weight_ffw_prev); gpu_context->h_weight_ffw_prev = nullptr;
		free(gpu_context->h_weight_ffw_prev_bias); gpu_context->h_weight_ffw_prev_bias = nullptr;
		free(gpu_context->h_weight_ffw_post_splitted); gpu_context->h_weight_ffw_post_splitted = nullptr;
		free(gpu_context->h_weight_ffw_post_bias_splitted); gpu_context->h_weight_ffw_post_bias_splitted = nullptr;
	}

	for (int batch_idx = 0; batch_idx < bert_state->num_batch; batch_idx++) {
		cudaFreeHost(gpu_context->h_buf_query[batch_idx]);
		cudaFreeHost(gpu_context->h_buf_key[batch_idx]);
		cudaFreeHost(gpu_context->h_buf_value[batch_idx]);
		cudaFreeHost(gpu_context->h_buf_score[batch_idx]);
		cudaFreeHost(gpu_context->h_buf_expsum[batch_idx]);
		cudaFreeHost(gpu_context->h_buf_softmax[batch_idx]);
		cudaFreeHost(gpu_context->h_buf_att_fc_result_split[batch_idx]);
		cudaFreeHost(gpu_context->h_buf_ffw_intermediate[batch_idx]);
		cudaFreeHost(gpu_context->h_buf_ffw_gelu[batch_idx]);
		cudaFreeHost(gpu_context->h_buf_ffw_result_split[batch_idx]);

		cudaFreeHost(gpu_context->h_buf_att[batch_idx]);
		cudaFreeHost(gpu_context->h_buf_att_layernorm[batch_idx]);
		cudaFreeHost(gpu_context->h_buf_ffw_layernorm[batch_idx]);
//		cudaFreeHost(gpu_context->h_buf_layernorm_mean[batch_idx]);
//		cudaFreeHost(gpu_context->h_buf_layernorm_tmp[batch_idx]);
		cudaFreeHost(gpu_context->h_buf_layernorm_nrm_v[batch_idx]);
	}
	free(gpu_context->h_buf_query); gpu_context->h_buf_query = nullptr;
	free(gpu_context->h_buf_key); gpu_context->h_buf_key = nullptr;
	free(gpu_context->h_buf_value); gpu_context->h_buf_value = nullptr;
	free(gpu_context->h_buf_score); gpu_context->h_buf_score = nullptr;
	free(gpu_context->h_buf_expsum); gpu_context->h_buf_expsum = nullptr;
	free(gpu_context->h_buf_softmax); gpu_context->h_buf_softmax = nullptr;
	free(gpu_context->h_buf_att); gpu_context->h_buf_att = nullptr;
	free(gpu_context->h_buf_att_fc_result_split); gpu_context->h_buf_att_fc_result_split = nullptr;
	free(gpu_context->h_buf_att_layernorm); gpu_context->h_buf_att_layernorm = nullptr;
	free(gpu_context->h_buf_ffw_intermediate); gpu_context->h_buf_ffw_intermediate = nullptr;
	free(gpu_context->h_buf_ffw_gelu); gpu_context->h_buf_ffw_gelu = nullptr;
	free(gpu_context->h_buf_ffw_result_split); gpu_context->h_buf_ffw_result_split = nullptr;
	free(gpu_context->h_buf_ffw_layernorm); gpu_context->h_buf_att_layernorm = nullptr;
//	free(gpu_context->h_buf_layernorm_mean); gpu_context->h_buf_layernorm_mean = nullptr;
//	free(gpu_context->h_buf_layernorm_tmp); gpu_context->h_buf_layernorm_tmp = nullptr;
	free(gpu_context->h_buf_layernorm_nrm_v); gpu_context->h_buf_layernorm_nrm_v = nullptr;
}

void cuda_multi_dev_context_deinit(BERT_State *bert_state, gpu_cuda_context_t *gpu_context) {
	// Destroy CUDA streams, cublas contexts
	cublasDestroy(gpu_context->cublas_handles[0]);
	free(gpu_context->cublas_handles);

	/// GPU memory
//	cudaFree(gpu_context->d_onevec); gpu_context->d_onevec = nullptr;
//	cudaFree(gpu_context->d_onemat); gpu_context->d_onemat = nullptr;

	if (gpu_context->d_attention_mask) {
		for (int batch_idx = 0; batch_idx < bert_state->num_batch; batch_idx++) {
			if (gpu_context->d_attention_mask[batch_idx]) cudaFree(gpu_context->d_attention_mask[batch_idx]);
		}
		free(gpu_context->d_attention_mask); gpu_context->d_attention_mask = nullptr;
	}

	if (gpu_context->d_input) {
		for (int batch_idx = 0; batch_idx < bert_state->num_batch; batch_idx++) {
			if (gpu_context->d_input[batch_idx]) cudaFree(gpu_context->d_input[batch_idx]);
		}
		free(gpu_context->d_input); gpu_context->d_input = nullptr;
	}

	if (gpu_context->d_weight) {
		for (int layer_idx = 0; layer_idx < bert_state->num_layer; layer_idx++) {
			for (int weight_idx = 0; weight_idx < WEIGHT_MAX_NUM; weight_idx++) {
				if (gpu_context->d_weight[layer_idx][weight_idx]) cudaFree(gpu_context->d_weight[layer_idx][weight_idx]);
			}
			free(gpu_context->d_weight[layer_idx]);

			cudaFree(gpu_context->d_weight_attention_fc_splitted[layer_idx]);
			cudaFree(gpu_context->d_weight_attention_fc_bias_splitted[layer_idx]);

			cudaFree(gpu_context->d_weight_ffw_prev[layer_idx]);
			cudaFree(gpu_context->d_weight_ffw_prev_bias[layer_idx]);
			cudaFree(gpu_context->d_weight_ffw_post_splitted[layer_idx]);
			cudaFree(gpu_context->d_weight_ffw_post_bias_splitted[layer_idx]);
		}
		free(gpu_context->d_weight); gpu_context->d_weight = nullptr;
		free(gpu_context->d_weight_attention_fc_splitted); gpu_context->d_weight_attention_fc_splitted = nullptr;
		free(gpu_context->d_weight_attention_fc_bias_splitted); gpu_context->d_weight_attention_fc_bias_splitted = nullptr;
		free(gpu_context->d_weight_ffw_prev); gpu_context->d_weight_ffw_prev = nullptr;
		free(gpu_context->d_weight_ffw_prev_bias); gpu_context->d_weight_ffw_prev_bias = nullptr;
		free(gpu_context->d_weight_ffw_post_splitted); gpu_context->h_weight_ffw_post_splitted = nullptr;
		free(gpu_context->d_weight_ffw_post_bias_splitted); gpu_context->h_weight_ffw_post_bias_splitted = nullptr;
	}

	for (int batch_idx = 0; batch_idx < bert_state->num_batch; batch_idx++) {
		cudaFree(gpu_context->d_buf_query[batch_idx]);
		cudaFree(gpu_context->d_buf_key[batch_idx]);
		cudaFree(gpu_context->d_buf_value[batch_idx]);
		cudaFree(gpu_context->d_buf_score[batch_idx]);
		cudaFree(gpu_context->d_buf_expsum[batch_idx]);
		cudaFree(gpu_context->d_buf_softmax[batch_idx]);
		cudaFree(gpu_context->d_buf_att_fc_result_split[batch_idx]);
		cudaFree(gpu_context->d_buf_ffw_intermediate[batch_idx]);
		cudaFree(gpu_context->d_buf_ffw_gelu[batch_idx]);
		cudaFree(gpu_context->d_buf_ffw_result_split[batch_idx]);

		cudaFree(gpu_context->d_buf_att[batch_idx]);
		cudaFree(gpu_context->d_buf_att_layernorm[batch_idx]);
		cudaFree(gpu_context->d_buf_ffw_layernorm[batch_idx]);
//		cudaFree(gpu_context->d_buf_layernorm_mean[batch_idx]);
//		cudaFree(gpu_context->d_buf_layernorm_tmp[batch_idx]);
		cudaFree(gpu_context->d_buf_layernorm_nrm_v[batch_idx]);
	}
	free(gpu_context->d_buf_query); gpu_context->d_buf_query = nullptr;
	free(gpu_context->d_buf_key); gpu_context->d_buf_key = nullptr;
	free(gpu_context->d_buf_value); gpu_context->d_buf_value = nullptr;
	free(gpu_context->d_buf_score); gpu_context->d_buf_score = nullptr;
	free(gpu_context->d_buf_expsum); gpu_context->d_buf_expsum = nullptr;
	free(gpu_context->d_buf_softmax); gpu_context->d_buf_softmax = nullptr;
	free(gpu_context->d_buf_att); gpu_context->d_buf_att = nullptr;
	free(gpu_context->d_buf_att_fc_result_split); gpu_context->d_buf_att_fc_result_split = nullptr;
	free(gpu_context->d_buf_att_layernorm); gpu_context->d_buf_att_layernorm = nullptr;
	free(gpu_context->d_buf_ffw_intermediate); gpu_context->d_buf_ffw_intermediate = nullptr;
	free(gpu_context->d_buf_ffw_gelu); gpu_context->d_buf_ffw_gelu = nullptr;
	free(gpu_context->d_buf_ffw_result_split); gpu_context->d_buf_ffw_result_split = nullptr;
	free(gpu_context->d_buf_ffw_layernorm); gpu_context->d_buf_ffw_layernorm = nullptr;
//	free(gpu_context->d_buf_layernorm_mean); gpu_context->d_buf_layernorm_mean = nullptr;
//	free(gpu_context->d_buf_layernorm_tmp); gpu_context->d_buf_layernorm_tmp = nullptr;
	free(gpu_context->d_buf_layernorm_nrm_v); gpu_context->d_buf_layernorm_nrm_v = nullptr;
}