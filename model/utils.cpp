#include <cstdio>
#include <cassert>
#include <cmath>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <memory.h>

#include "utils.hpp"

float diff(float *a, float *b, int s0) {
	float ret=0;
	for(int i=0;i<s0;i++){
		if(a[i]-b[i]>0.1 || a[i]-b[i]<-0.1 || std::isnan(a[i]-b[i])) {
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
	       "%s [-t num_th_head:default=1] [-b num_th_batch:default=1] [-m mode: default=0] vocab_sz token_sz num_batch num_head seq_len hidden_sz ff_sz num_layer\n"
	       "\tvocab_sz: # of vocabs in dictionary\n"
	       "\ttoken_sz: the size of token\n"
	       "\tnum_batch: the number of batches\n"
	       "\tnum_head: multi-head self attention\n"
	       "\tseq_len: sequence length\n"
	       "\thidden_sz: hidden vector size\n"
	       "\tff_sz: size for feedforward network\n"
	       "\tnum_layer: the number of layers\n"
	       "\tOption:\n"
	       "\t\tt: the number of mkl_thread\n"
	       "\t\tb: the number of batch_thread\n"
	       "2. Verification Mode\n"
	       "%s [-t num_th_head:default=1] [-b num_th_batch:default=1] [-m mode:default=0] dir_smallest dir_weight\n"
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

	/// # of threads
	if (params->num_th_batch < 1 || params->num_th_batch > 48 || (params->num_batch % params->num_th_batch)) {
		printf("[ERR] batch_thread number (%d) should be divisor of the BATCH (%d)\n", params->num_th_batch, params->num_batch);
		return -1;
	}
	if (params->num_th_head < 1 || params->num_th_head > 48) {
		printf("[ERR] head_thread number (%d) should be 1 to 48\n", params->num_th_head);
		return -1;
	}
	if (params->num_th_batch * params->num_th_head > 48) {
		printf("[ERR] Too many threads!!! (our machine supports up to 48 cores) (batch: %d, head: %d\n", params->num_th_batch, params->num_th_head);
		return -1;
	}

	/// check params for performance mode
	if (params->execution_mode == EXEC_MODE_PERF_TEST) {
		if (params->hidden_size % params->num_heads) {
			printf("[ERR] num heads (%d) should be devisor of the hidden_size (%d)\n", params->num_heads, params->hidden_size);
			return -1;
		}

		if (params->feedforwardsize % params->num_heads) {
			printf("[ERR] num heads (%d) should be devisor of the feedforward_size (%d)\n", params->num_heads, params->feedforwardsize);
			return -1;
		}

		if (params->num_heads % params->num_th_head) {
			printf("[ERR] num_th_head (%d) should be devisor of the num_heads (%d)\n", params->num_th_head, params->num_heads);
			return -1;
		}

		if (params->num_batch % params->num_th_batch) {
			printf("[ERR] num_th_batch (%d) should be devisor of the num_batch (%d)\n", params->num_th_batch, params->num_batch);
			return -1;
		}

		if (params->column_based) {
			if (params->seq_length % params->chunk_size) {
				printf("[ERR] chunk_size (%d) should be devisor of the seq_length (%d)\n", params->chunk_size, params->seq_length);
				return -1;
			}
		}
	}

	return 0;
}

void parse_args(int argc, char *argv[], Params *params) {
	// Default Parameters
	int opt, optcount = 0;
	params->debug_flag = false;
	params->execution_mode = EXEC_MODE_PERF_TEST;
	params->num_th_head = 1;
	params->num_th_batch = 1;

	while ((opt = getopt(argc, argv, "dht:b:m:c:")) != -1){
		optcount++;
		switch (opt) {
			case 'd':
				params->debug_flag = true;
				break;
			case 'h':
				print_usage(argc, argv);
				exit(0);
			case 't':
				params->num_th_head = atoi(optarg);
				if (params->num_th_head < 1){
					fprintf(stderr, "num_th_head should not be less than 1:"
					                "cur_val %d\n.", params->num_th_head);
					print_usage(argc, argv);
					exit(1);
				}
				break;
			case 'b':
				params->num_th_batch = atoi(optarg);
				if (params->num_th_batch < 1){
					fprintf(stderr, "num_th_batch should not be less than 1:"
					                "cur_val %d\n.", params->num_th_batch);
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
			case 'c':
				params->column_based = true;
				params->chunk_size = atoi(optarg);
				if ((params->chunk_size < 1)) {
					fprintf(stderr, "Invalid chunk_size (%d)\n", params->chunk_size);
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
	       "num_th_head %d\n"
	       "num_th_batch %d\n"
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
	       params->num_th_head,
	       params->num_th_batch,
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

#ifdef CLFLUSH
//#define FLUSH_TLB_CMD  _IOWR('b', 1, unsigned long)
//static int ioctlfd = 0;

void flush_cache(void* data, unsigned long size) {
	//static const int data_spray_entries = 1024 * 1024 * 10; // 40MB
	//static int volatile data_spray [sizeof(int) * data_spray_entries];
	//int volatile temp = 0;
	//for (int i = 0; i < data_spray_entries; i += 1){
		//temp += data_spray[i];
		//data_spray[i] = temp;
		////asm volatile("mfence":::"memory");
	//}
	//asm volatile("mfence":::"memory");
	
	char *c_data = (char*)data;
	const int CL_SIZE = 32;
	for (long i = 0; i < size; i += CL_SIZE){
		char* p = c_data + i;
		asm volatile ("clflush %0" :: "m" (*(char*) p) : "memory");
	}
	asm volatile("mfence":::"memory");

	// tlb flush
	//if (!ioctlfd) {
		//ioctlfd = open("/dev/flush_tlb", O_RDWR);
		//if (ioctlfd < 0) {
			//perror("Open flush_tlb binder failed");
			//exit(1);
		//}
	//}

	//unsigned long addr = 0;
	//if (ioctl(ioctlfd, FLUSH_TLB_CMD, &addr)) {
		//perror("why");
		//exit(1);
	//}
}
#else
void flush_cache(void* data, unsigned long size) {asm volatile("mfence":::"memory");}
#endif

#ifdef PREFETCH
void prefetch_data(void *data, unsigned long size) {
	volatile char c_tmp = 0;
	char *c_data = (char*)data;
	/// If flush with multi-threaded config, threads flush other thread's data
//	const int CL_SIZE = 32;
//
//	/// Flush it first
//	for (long i = 0; i < size; i += CL_SIZE){
//		char* p = c_data + i;
//		asm volatile ("clflush %0" :: "m" (*(char*) p) : "memory");
//	}
//	asm volatile("mfence":::"memory");

	/// Load data
	for (long i = 0; i<size; i++) {
		c_tmp += c_data[i];
	}
}
#else
void prefetch_data(void *data, unsigned long size) {}
#endif
