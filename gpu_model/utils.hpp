#ifndef GPU_MODEL_UTILS_HPP
#define GPU_MODEL_UTILS_HPP
#include <string>

typedef enum {
	EXEC_MODE_PERF_TEST = 0,
	EXEC_MODE_VERIFICATION,
	EXEC_MODE_MAX,
} EXEC_MODE;

typedef enum {
	MEMCPY_MODE_ALL_OVERHEAD = 0,
	MEMCPY_MODE_SYNC_OVERHEAD,
	MEMCPY_MODE_NO_ALL_OVERHEAD,
} MEMCPY_MODE;

struct Params {
	bool debug_flag;
	MEMCPY_MODE memcpy_mode;

	EXEC_MODE execution_mode;
	int gpu_id;
	int num_streams;
	int thread_block_size;
	int num_gpus;

	// performance mode specific params
	int vocab_size;
	int token_size;
	int num_batch;
	int num_heads;
	int seq_length;
	int hidden_size;
	int feedforwardsize;
	int num_layer;

	char *dir_random_chunk;

	// verification mode specific params
	std::string dir_smallest;
	std::string dir_weight;
};

//
// Utility functions
//
float diff(float *a, float *b, int s0);

void parse_args(int argc, char *argv[], Params *params);
int  param_sanity_check(Params *params);
void dump_params(Params* params);

float *mat_rand_float(long m, long n, float min_val = -1., float max_val = 1.);
int* mat_rand_int(long m, long n, int min_val = -1, int max_val = 1);

void matcopy_row_to_col_float(float *dst, float *src, int M, int N);
void matcopy_col_to_row_float(float *dst, float *src, int M, int N);
float *vec_to_mat_span_float(float *srcvec, int M, int N);
#endif //GPU_MODEL_UTILS_HPP