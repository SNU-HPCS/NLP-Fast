#ifndef LOG_HPP
#define LOG_HPP
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <chrono>

#include "enum_util.hpp"
#include "cachestat_monitor.hpp"
#include "cachestat.hpp"

struct LatCounter {
	std::chrono::system_clock::time_point ts_begin;
	std::chrono::system_clock::time_point ts_end;

	void begin() {
		ts_begin = std::chrono::system_clock::now();
	}
	void end() {
		ts_end = std::chrono::system_clock::now();
	}
	double get_latency() {
		auto dur = std::chrono::duration<double> (ts_end - ts_begin);
		return dur.count();
	}
};

void mark_time(const char* msg);

/////////////////////////////
////////// Logger ///////////
/////////////////////////////
#define _STAT_TYPE_EMBEDDING(GEN) \
	GEN(EMBED_LOOKUP) \
	GEN(EMBED_POSTPROCESSING) \
	GEN(EMBED_MAX)

#define _STAT_TYPE_TRANSFORMER_ENCODER(GEN) \
	GEN(ATTENTION_LAYER_MASK_CREATION) \
	GEN(ATTENTION_LAYER_Q_GEN) \
	GEN(ATTENTION_LAYER_K_GEN) \
	GEN(ATTENTION_LAYER_SCORE_CAL) \
	GEN(ATTENTION_LAYER_SCORE_NORM) \
	GEN(ATTENTION_LAYER_MASK_SUB) \
	GEN(ATTENTION_LAYER_SOFTMAX) \
	GEN(ATTENTION_LAYER_V_GEN) \
	GEN(ATTENTION_LAYER_WEIGHTED_SUM) \
	GEN(ATTENTION_LAYER_HEAD_MERGE) \
	GEN(ATTENTION_FC) \
	GEN(ATTENTION_RESIDUAL_CONNECT) \
	GEN(ATTENTION_LAYERNORM) \
	GEN(FEEDFORWARD_PRE) \
	GEN(FEEDFORWARD_GELU) \
	GEN(FEEDFORWARD_POST) \
	GEN(FEEDFORWARD_RESIDUAL_CONNECT) \
	GEN(FEEDFORWARD_LAYERNORM) \
	GEN(TRANSFORMER_ENCODER_MAX)

enum STAT_TYPE_EMBEDDING_ENUM {
	_STAT_TYPE_EMBEDDING(GEN_ENUM)
};
static const char *STAT_TYPE_EMBEDDING_STR[] = {
		_STAT_TYPE_EMBEDDING(GEN_STR)
};

enum STAT_TYPE_TRANSFORMER_ENCODER_ENUM {
	_STAT_TYPE_TRANSFORMER_ENCODER(GEN_ENUM)
};
static const char *STAT_TYPE_TRANSFORMER_ENCODER_STR[] = {
		_STAT_TYPE_TRANSFORMER_ENCODER(GEN_STR)
};

class Logger {
	int batch_tid;
	int head_tid;
//	int batch_idx;  // Index in a batch
	int entry_num;
	static int num_batchs;
	static int num_heads;
	static int batch_thread_num;
	static int head_thread_num;
	static int tot_thread_num; // Total number of threads (batch_threads * (head_threads + 1))
	static int pmu_size;

	LatCounter *lat_counter;
	static CacheStatMonitor **csmon_arr;
	CacheStatMonitor::CacheStatCounter *cs_counter;

	double *latency_arr;
	CacheStat *cachestat_arr;

public:
	Logger() {}
	void init(int transformer_layer_num);
	void set_tids(int _batch_tid, int _head_tid) { batch_tid = _batch_tid; head_tid = _head_tid; }
	int get_batch_tid() { return batch_tid; }
	int get_head_tid() { return head_tid; }
	void Logger_deinit();

	static void set_num_batchs(int _num_batchs) { num_batchs = _num_batchs; }
	static void set_num_heads(int _num_heads) { num_heads = _num_heads; }
	// [head_idx == -1] => logger for batch thread | [head_idx >= 0] => logger for head thread
	static int get_logger_idx(int batch_idx, int head_idx, bool is_partial_fc = false) {
		if (is_partial_fc) {
			return batch_idx * (num_heads * 2 + 1) + head_idx + 1;
		} else {
			return batch_idx * (num_heads + 1) + head_idx + 1;
		}
	}
	// [head_tid == -1] => batch thread | [head_tid >= 0] => head thread
	static int get_thread_idx(int batch_tid, int head_tid, bool is_partial_fc) {
		if (is_partial_fc) {
			return batch_tid * (head_thread_num * 2 + 1) + head_tid + 1;
		} else {
			return batch_tid * (head_thread_num + 1) + head_tid + 1;
		}
	}
	static void init_csmon_arr(int batch_thread_num, int head_thread_num, bool is_partial_fc = false);
	static void init_csmon(int batch_tid, int head_tid, bool inherit, bool is_partial_fc = false);
	static void deinit_csmon(int batch_tid, int head_tid, bool is_partial_fc = false);
	static void csmon_start(int batch_tid, int head_tid, bool is_partial_fc = false) { csmon_arr[get_thread_idx(batch_tid, head_tid, is_partial_fc)]->start(false); }
	static void csmon_stop(int batch_tid, int head_tid, bool is_partial_fc = false) { csmon_arr[get_thread_idx(batch_tid, head_tid, is_partial_fc)]->stop(); }
	static CacheStatMonitor *get_csmon(int batch_tid, int head_tid, bool is_partial_fc = false) { return csmon_arr[get_thread_idx(batch_tid, head_tid, is_partial_fc)]; }
	void set_cs_counter(CacheStatMonitor::CacheStatCounter *_cs_counter) { cs_counter = _cs_counter; }
	void free_cs_counter() { delete(cs_counter); cs_counter = nullptr; }
	void embed_logging_begin(STAT_TYPE_EMBEDDING_ENUM embed_subtype);
	void embed_logging_end(STAT_TYPE_EMBEDDING_ENUM embed_subtype);
	void infer_logging_begin(STAT_TYPE_TRANSFORMER_ENCODER_ENUM tf_enc_subtype, int layer_idx);
	void infer_logging_end(STAT_TYPE_TRANSFORMER_ENCODER_ENUM tf_enc_subtype, int layer_idx);
	double get_infer_latency(STAT_TYPE_TRANSFORMER_ENCODER_ENUM tf_enc_subtype, int layer_idx);

	void dump_latency();
	void dump_cs();
};

#endif
