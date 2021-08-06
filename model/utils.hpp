#ifndef MODEL_UTILS_HPP
#define MODEL_UTILS_HPP
#include <string>
#include <cstdio>
#include <unordered_map>

typedef enum {
	EXEC_MODE_PERF_TEST = 0,
	EXEC_MODE_VERIFICATION,
	EXEC_MODE_MAX,
} EXEC_MODE;

struct Params {
	bool debug_flag;

	EXEC_MODE execution_mode;
	int num_th_head;             // the number of threads for BLAS
	int num_th_batch;           // the number of threads for pthread (unit: batch)
	bool column_based;
	int chunk_size;

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
void flush_cache(void *data, unsigned long size);
void prefetch_data(void *data, unsigned long size);


#endif //MODEL_UTILS_HPP
