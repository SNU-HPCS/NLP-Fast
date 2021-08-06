#include <chrono>
#include "log.hpp"
#include "cachestat.hpp"
#include "cachestat_monitor.hpp"

void mark_time(const char* msg){
	using std::chrono::system_clock;
	static system_clock::time_point start_t = system_clock::now();
	static system_clock::time_point prev_t = system_clock::now();
	system_clock::time_point cur_t = system_clock::now();
	auto s = std::chrono::duration<double> (cur_t - start_t);
	auto s2 = std::chrono::duration<double> (cur_t - prev_t);
	printf("%20s: %fs %fs\n", msg, s.count(), s2.count());
	prev_t = cur_t;
}


int Logger::num_batchs = 0;
int Logger::num_heads = 0;
int Logger::batch_thread_num = 0;
int Logger::head_thread_num = 0;
int Logger::tot_thread_num = 0;
int Logger::pmu_size = 0;
CacheStatMonitor **Logger::csmon_arr = nullptr;

void Logger::init(int transformer_layer_num) {
	entry_num = EMBED_MAX + TRANSFORMER_ENCODER_MAX * transformer_layer_num;

	/// Latency Counter
	lat_counter = new LatCounter[entry_num];
	memset(lat_counter, 0, sizeof(double) * entry_num);

	/// Latency & Cache data array
	latency_arr = new double[entry_num];
	cachestat_arr = new CacheStat[entry_num];
	memset(latency_arr, 0, sizeof(double) * entry_num);
	memset(cachestat_arr, 0 , sizeof(CacheStat) * entry_num);
}

void Logger::Logger_deinit() {
	delete [] lat_counter;
	if (Logger::csmon_arr) {
		for (int i = 0; i < Logger::tot_thread_num; i++) {
			assert(Logger::csmon_arr[i] == nullptr);
		}
		delete [] Logger::csmon_arr;
		Logger::csmon_arr = nullptr;
	}
	delete cs_counter;
	delete [] latency_arr;
	delete [] cachestat_arr;
}

void Logger::init_csmon_arr(int _batch_thread_num, int _head_thread_num, bool is_partial_fc) {
	batch_thread_num = _batch_thread_num;
	head_thread_num = _head_thread_num;
	if (is_partial_fc) {
		tot_thread_num = batch_thread_num * (head_thread_num * 2 + 1);
	} else {
		tot_thread_num = batch_thread_num * (head_thread_num + 1);
	}
	csmon_arr = new CacheStatMonitor*[tot_thread_num];
	memset(csmon_arr, 0, sizeof(CacheStatMonitor*) * tot_thread_num);
}

/**
 * init cache stat monitor
 * @param batch_tid : thread idx of batch threads
 * @param head_tid : [if >= 0] thread idx of head threads | [if == -1] batch thread (not head thread)
 */
void Logger::init_csmon(int batch_tid, int head_tid, bool inherit, bool is_partial_fc) {
	int thread_idx = get_thread_idx(batch_tid, head_tid, is_partial_fc);
	/// Only initialize cache monitor at once (per thread)
	if (Logger::csmon_arr[thread_idx] == nullptr) {
		int _pmu_size;
		const char **pmu = select_pmu("default", _pmu_size); // Todo: get pmu_type from a command argument
		Logger::csmon_arr[thread_idx] = new CacheStatMonitor(inherit, pmu, _pmu_size);
		Logger::pmu_size = _pmu_size;
	} else {
		assert(0);
	}
}

void Logger::deinit_csmon(int batch_tid, int head_tid, bool is_partial_fc) {
	int thread_idx = get_thread_idx(batch_tid, head_tid, is_partial_fc);
	/// Only initialize cache monitor at once (per thread)
	assert(Logger::csmon_arr[thread_idx]);
	delete Logger::csmon_arr[thread_idx];
	Logger::csmon_arr[thread_idx] = nullptr;
}

void Logger::embed_logging_begin(STAT_TYPE_EMBEDDING_ENUM embed_subtype) {
	lat_counter[embed_subtype].begin();
	cs_counter->begin();
}
void Logger::embed_logging_end(STAT_TYPE_EMBEDDING_ENUM embed_subtype) {
	lat_counter[embed_subtype].end();
	latency_arr[embed_subtype] += lat_counter[embed_subtype].get_latency();
	cachestat_arr[embed_subtype] += cs_counter->end();
}
void Logger::infer_logging_begin(STAT_TYPE_TRANSFORMER_ENCODER_ENUM tf_enc_subtype, int layer_idx) {
	int idx = EMBED_MAX + layer_idx * TRANSFORMER_ENCODER_MAX + tf_enc_subtype;
	cs_counter->begin();
	lat_counter[idx].begin();
}
void Logger::infer_logging_end(STAT_TYPE_TRANSFORMER_ENCODER_ENUM tf_enc_subtype, int layer_idx) {
	int idx = EMBED_MAX + layer_idx * TRANSFORMER_ENCODER_MAX + tf_enc_subtype;
	lat_counter[idx].end();
	latency_arr[idx] += lat_counter[idx].get_latency();
	cachestat_arr[idx] += cs_counter->end();
}
double Logger::get_infer_latency(STAT_TYPE_TRANSFORMER_ENCODER_ENUM tf_enc_subtype, int layer_idx) {
	int idx = EMBED_MAX + layer_idx * TRANSFORMER_ENCODER_MAX + tf_enc_subtype;
	return lat_counter[idx].get_latency();
}

#define NAME_BUF_LEN 256

void Logger::dump_latency() {
	const char *name = nullptr;
	char name_buf[NAME_BUF_LEN];

	for (int entry_idx = 0; entry_idx < entry_num; entry_idx++) {
		if (entry_idx < EMBED_MAX) {
			name = STAT_TYPE_EMBEDDING_STR[entry_idx];
			snprintf(name_buf, NAME_BUF_LEN, "%s", name);
		} else {
			name = STAT_TYPE_TRANSFORMER_ENCODER_STR[(entry_idx - EMBED_MAX) % TRANSFORMER_ENCODER_MAX];
			snprintf(name_buf, NAME_BUF_LEN, "LAYER-%d-%s", ((entry_idx - EMBED_MAX) / TRANSFORMER_ENCODER_MAX), name);
		}

		printf("%s: %lf\n", name_buf, latency_arr[entry_idx]);
	}
}

void Logger::dump_cs() {
	const char *name = nullptr;
	CacheStat *cachestat = nullptr;
	char name_buf[NAME_BUF_LEN];

	for (int i = 0; i < CacheStat::NUM_CS_DATA; ++i) {
		printf(" %s", CacheStat::STR_ENUM_CS_DATA[i]);
	}
	printf("\n");

	for (int entry_idx = 0; entry_idx < entry_num; entry_idx++) {
		cachestat = &cachestat_arr[entry_idx];

		if (entry_idx < EMBED_MAX) {
			name = STAT_TYPE_EMBEDDING_STR[entry_idx];
			snprintf(name_buf, NAME_BUF_LEN, "%s", name);
		} else {
			name = STAT_TYPE_TRANSFORMER_ENCODER_STR[(entry_idx - EMBED_MAX) % TRANSFORMER_ENCODER_MAX];
			snprintf(name_buf, NAME_BUF_LEN, "LAYER-%d-%s", ((entry_idx - EMBED_MAX) / TRANSFORMER_ENCODER_MAX), name);
		}

		printf("%s: ",name_buf);
		for (int i = 0; i < Logger::pmu_size; i++) {
			printf("%lu (%lf) ", cachestat->data[i].value,
					((double) cachestat->data[i].time_running) / ((double) cachestat->data[i].time_enabled) * 100.0);
		}
		printf("\n");
	}
}