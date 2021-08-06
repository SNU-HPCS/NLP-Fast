#include <iostream>
#include <cmath>
#include <mkl.h>
#include <sys/sysinfo.h>

#include "utils.hpp"
#ifdef BASELINE
#include "baseline_model.hpp"
#elif PARTIAL_HEAD
#include "partial_head_model.hpp"
#elif COLUMN
#include "column_model.hpp"
#elif ALL_OPT
#include "all_opt_model.hpp"
#else
#endif
#include "log.hpp"
#include "bert_state.hpp"

using namespace std;

int main(int argc, char *argv[]){
	float *ret = nullptr, *ref = nullptr;
	BERT_State *bert_state = nullptr;
	Params params = {};
	Logger *loggers = nullptr;
	int dump_head_num;
	bool is_partial_fc;

	// Force MKL to use the given number of threads via mkl_set_num_threads function.
	// If DYNAMIC mode enables, MKL may use less number of threads given by mkl_set_num_threads
	//mkl_set_dynamic(0);
	//mkl_set_num_threads(1);

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
	mark_time("<Init_BERT_State> END");

	// Init BERT object
	BERT bert=BERT(&params, bert_state, false, true, true, true);
	mark_time("<Init_BERT> END");

	// Init logger
	Logger::set_num_batchs(bert_state->num_batch);
	Logger::set_num_heads(bert_state->num_heads);
#ifdef ALL_OPT
	dump_head_num = bert_state->num_heads * 2;
	is_partial_fc = true;
#else
	dump_head_num = bert_state->num_heads;
	is_partial_fc = false;
#endif
	loggers = new Logger[bert_state->num_batch * (dump_head_num + 1)];
	for (int i = 0; i < bert_state->num_batch * (dump_head_num + 1); i++) {
		loggers[i].init(bert_state->num_layer);
	}
	for (int i = 0; i < params.num_th_batch * params.num_th_head; i++) {
		bert_state->att_st_list[i]->loggers = loggers;
		bert_state->ffw_st_list[i]->loggers = loggers;
	}

	/// Start execution!!!
	ret = (float*)bert.forward(bert_state->input, bert_state->mask, bert_state->token_type, loggers);
	mark_time("<BERT> END");

	/// Verification (verification mode only)
	if (params.execution_mode == EXEC_MODE_VERIFICATION) {
		/// For debugging purpose
		bert.dump_values();

		ref = load_pooled_output(&params, bert_state);    // Extract answer
		cout<<"difference, "<<diff(ret, ref, bert_state->num_batch * bert_state->hidden_size)<<endl;
	} else if (params.execution_mode == EXEC_MODE_PERF_TEST) {
		/// Dump latency & cachestat
		for (int batch_idx = 0; batch_idx < bert_state->num_batch; batch_idx++) {
			for (int head_idx = -1; head_idx < dump_head_num; head_idx++) {
				printf("===== CacheStat (batch_idx: %d, batch_tid: %d, head_idx: %d, head_tid: %d) =====\n",
						batch_idx, loggers[Logger::get_logger_idx(batch_idx, head_idx, is_partial_fc)].get_batch_tid(),
						head_idx, loggers[Logger::get_logger_idx(batch_idx, head_idx, is_partial_fc)].get_head_tid());
				loggers[Logger::get_logger_idx(batch_idx, head_idx, is_partial_fc)].dump_cs();
				printf("===== CacheStat DONE =====\n");

				printf("===== LatencyStat (batch_idx: %d, batch_tid: %d, head_idx: %d, head_tid: %d) =====\n",
						batch_idx, loggers[Logger::get_logger_idx(batch_idx, head_idx, is_partial_fc)].get_batch_tid(),
						head_idx, loggers[Logger::get_logger_idx(batch_idx, head_idx, is_partial_fc)].get_head_tid());
				loggers[Logger::get_logger_idx(batch_idx, head_idx, is_partial_fc)].dump_latency();
				printf("===== LatencyStat DONE =====\n");
			}
		}
	}

	// Deinit (bert, logger, bert_state)
	bert.BERT_deinit();
	for (int i = 0; i < bert_state->num_batch * (dump_head_num + 1); i++) {
		loggers[i].Logger_deinit();
	}
	delete [] loggers;
	free(params.dir_random_chunk);
	deinit_bert_state(&params, bert_state);
	free(bert_state);

	return 0;
}
