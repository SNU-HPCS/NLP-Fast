#include <cstdio>
#include <cassert>
#include <cmath>
#include <iostream>
#include <unistd.h>
#include <memory.h>
#include <random>

#include "utils.hpp"

float diff(float *a, float *b, int s0) {
	float ret=0;
	for(int i=0;i<s0;i++){
		if(a[i]-b[i]>0.1f || a[i]-b[i]<-0.1f || std::isnan(a[i]-b[i])) {
			std::cout<<i<<", "<<a[i]<<", "<<b[i]<<std::endl;
		}
		ret+=(a[i]-b[i])*(a[i]-b[i]);
	}
	return ret;
}

static void print_usage(int argc, char* argv[]) {
	printf("Current Command\n");
	for (int i = 0; i < argc; ++i){
		printf("%s ", argv[i]);
	}
	printf("\n1. Profiling Test\n"
	       "%s [-p memcpy_mode] [-s num_streams:default=1] [-g gpu_id:gpu_id=0] [-b thread_block_size: default=64] vocab_sz token_sz num_batch num_head seq_len hidden_sz ff_sz num_layer\n"
	       "\tvocab_sz: # of vocabs in dictionary\n"
	       "\ttoken_sz: the size of token\n"
	       "\tnum_batch: the number of batches\n"
	       "\tnum_head: multi-head self attention\n"
	       "\tseq_len: sequence length\n"
	       "\thidden_sz: hidden vector size\n"
	       "\tff_sz: size for feedforward network\n"
	       "\tnum_layer: the number of layers\n"
	       "\tOption:\n"
	       "\t\ts: the number of CUDA streams\n"
	       "\t\tg: GPU ID\n"
	       "\t\tb: thread_block_size (CUDA)\n"
	       "2. Verification Mode\n"
	       "%s [-s num_streams:default=1] [-g gpu_id:gpu_id=0] [-b thread_block_size: default=64] dir_smallest dir_weight\n"
	       "\tdir_smallest: directory containing input, mask, token_type\n"
	       "\tdir_weight: directory containing all weights\n"
	       "3. Help\n"
	       "%s -h\n", argv[0], argv[0], argv[0]);
}

static void check_narg(int nargs, int req_nargs) {
	if (nargs < req_nargs){
		printf("Positional argument is not enough (current: %d args). "
		       "Require %d arguments\n", nargs, req_nargs);
		exit(1);
	}
}

int param_sanity_check(Params *params) {
	int rc;

	/// check params for performance mode
	if (params->execution_mode == EXEC_MODE_PERF_TEST) {
		if (params->hidden_size % params->num_heads) {
			printf("[ERR] num heads (%d) should be devisor of the hidden_size (%d)\n", params->num_heads, params->hidden_size);
			return -1;
		}

		if (params->hidden_size % params->num_streams) {
			printf("[ERR] num_streams (%d) should be devisor of the hidden_size (%d)\n", params->num_streams, params->hidden_size);
			return -1;
		}

		if (params->num_heads % params->num_streams) {
			printf("[ERR] num_streams (%d) should be devisor of the num_heads (%d)\n", params->num_streams, params->num_heads);
			return -1;
		}

		if (params->feedforwardsize % params->num_heads) {
			printf("[ERR] num heads (%d) should be devisor of the feedforward_size (%d)\n", params->num_heads, params->feedforwardsize);
			return -1;
		}
	}

	return 0;
}

void parse_args(int argc, char *argv[], Params *params) {
	// Default Parameters
	int opt, optcount = 0, t_int;
	params->debug_flag = false;
	params->memcpy_mode = MEMCPY_MODE_ALL_OVERHEAD;
	params->execution_mode = EXEC_MODE_PERF_TEST;
	params->num_streams = 1;
	params->thread_block_size = 64;
	params->num_gpus = 1;

	while ((opt = getopt(argc, argv, "dhs:m:p:g:b:n:")) != -1){
		optcount++;
		switch (opt) {
			case 'd':
				params->debug_flag = true;
				break;
			case 'p':
				t_int = atoi(optarg);
				switch (t_int) {
					case 0:
						params->memcpy_mode = MEMCPY_MODE_ALL_OVERHEAD;
						break;
					case 1:
						params->memcpy_mode = MEMCPY_MODE_SYNC_OVERHEAD;
						break;
					case 2:
						params->memcpy_mode = MEMCPY_MODE_NO_ALL_OVERHEAD;
						break;
					default:
						fprintf(stderr, "memcpy_mode should be 0, 1, 2 (cur_val: %d)\n.", t_int);
						print_usage(argc, argv);
						exit(1);
				}
				break;
			case 'h':
				print_usage(argc, argv);
				exit(0);
			case 's':
				params->num_streams = atoi(optarg);
				if (params->num_streams < 1){
					fprintf(stderr, "num_streams should not be less than 1:"
					                "cur_val %d\n.", params->num_streams);
					print_usage(argc, argv);
					exit(1);
				}
				break;
			case 'm':
				params->execution_mode = (EXEC_MODE) atoi(optarg);
				if ((params->execution_mode < 0) || (params->execution_mode >= EXEC_MODE_MAX)) {
					fprintf(stderr, "Invalid execution mode\n");
					print_usage(argc, argv);
					exit(1);
				}
				break;
			case 'g':
				params->gpu_id = atoi(optarg);
				break;
			case 'b':
				params->thread_block_size = atoi(optarg);
				if ((params->thread_block_size < 1)) {
					fprintf(stderr, "Invalid thread_block_size (%d)\n", params->thread_block_size);
					print_usage(argc, argv);
					exit(1);
				}
				break;
			case 'n':
				params->num_gpus = atoi(optarg);
				if ((params->num_gpus < 1)) {
					fprintf(stderr, "Invalid num_gpus (%d)\n", params->num_gpus);
					print_usage(argc, argv);
					exit(1);
				}
				break;

			default:
				fprintf(stderr, "Unknown option: %c", opt);
				exit(1);
		}
	}

	if (params->execution_mode == EXEC_MODE_PERF_TEST) {  // profile mode
		check_narg(argc - optind, 8);
		params->vocab_size = atoi(argv[optind]);
		params->token_size = atoi(argv[optind+1]);
		params->num_batch = atoi(argv[optind+2]);
		params->num_heads = atoi(argv[optind+3]);
		params->seq_length = atoi(argv[optind+4]); // bool options
		params->hidden_size = atoi(argv[optind+5]);
		params->feedforwardsize = atoi(argv[optind+6]);
		params->num_layer = atoi(argv[optind+7]);

		if ((argc - optind) >= 9) {
			// optional argument (random_chunk_bin)
			params->dir_random_chunk = strdup(argv[optind+8]);
		} else {
			params->dir_random_chunk = strdup("");
		}

		dump_params(params);
	} else if (params->execution_mode == EXEC_MODE_VERIFICATION) {    // verification mode
		check_narg(argc - optind, 2);
		const char *dir_smallest = argv[optind];
		const char *dir_weight = argv[optind + 1];
		params->dir_smallest = std::string {dir_smallest};
		params->dir_weight = std::string {dir_weight};
	} else {
		assert(0);
	}
}

void dump_params(Params* params) {
	printf("============ Options ============\n");
	printf("debug_flag %d\n"
	       "exec_mode %s\n"
	       "num_streams %d\n"
	       "gpu_id %d\n"
	       "vocab_sz %d\n"
	       "token_sz %d\n"
	       "num_batch %d\n"
	       "num_head %d\n"
	       "seq_len %d\n"
	       "hidden_sz %d\n"
	       "ff_sz %d\n"
	       "num_layer %d\n",
	       params->debug_flag,
	       (params->execution_mode == EXEC_MODE_PERF_TEST) ? "PERF_TEST" : "VERIFICATION",
	       params->num_streams,
	       params->gpu_id,
	       params->vocab_size,
	       params->token_size,
	       params->num_batch,
	       params->num_heads,
	       params->seq_length,
	       params->hidden_size,
	       params->feedforwardsize,
	       params->num_layer);
	printf("=================================\n");
}

std::default_random_engine generator;
float* mat_rand_float(const long m, const long n, const float min_val, const float max_val) {
	// Temporal
	auto nelem = m * n;
	float *data = nullptr;
	if (nelem == 0)
		return nullptr;

	if (nelem / n != m){
		printf("<assert> m=%ld, n=%ld\n", m, n);
		assert(nelem > 0);
	}

	data = (float*)malloc(nelem * sizeof(float));
	std::uniform_real_distribution<float> distribution(min_val, max_val);
	for (long i = 0; i < nelem; ++i) {
		data[i] = static_cast<float>(distribution(generator));
	}
	return data;
}

int* mat_rand_int(const long m, const long n, const int min_val, const int max_val) {
	// Temporal
	auto nelem = m * n;
	int *data = nullptr;
	if (nelem == 0)
		return nullptr;

	if (nelem / n != m){
		printf("<assert> m=%ld, n=%ld\n", m, n);
		assert(nelem > 0);
	}

	data = (int*)malloc(nelem * sizeof(int));
	std::uniform_int_distribution<int> distribution(min_val, max_val - 1);
	for (long i = 0; i < nelem; ++i) {
		data[i] = static_cast<int>(distribution(generator));
	}
	return data;
}

/**
 *
 * @param dst : destination ptr (M x N) (Col-Major)
 * @param src : source ptr (M x N) (Row-Major)
 * @param M
 * @param N
 */
void matcopy_row_to_col_float(float *dst, float *src, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			dst[j * M + i] = src[i * N + j];
		}
	}
}
void matcopy_col_to_row_float(float *dst, float *src, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			dst[i * N + j] = src[j * M + i];
		}
	}
}

/**
 *
 * @param srcvec : size N vector
 * @param M
 * @param N
 * @return new M x N matrix, free srcvec
 */
float *vec_to_mat_span_float(float *srcvec, int M, int N) {
	float *new_matrix = (float *)malloc(sizeof(float) * M * N);

	for (int i = 0; i < M; i++) {
		memcpy(&new_matrix[N * i], srcvec, sizeof(float) * N);
	}
	free(srcvec);

	return new_matrix;
}