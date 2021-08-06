#ifndef MODEL_CACHESTAT_MONITOR_HPP
#define MODEL_CACHESTAT_MONITOR_HPP

#include <cstdint>
#include <linux/perf_event.h>
#include <string.h>
#include <stdio.h>
#include <perfmon/pfmlib_perf_event.h>
#include "cachestat.hpp"

static const char* DefaultPMU[] = {
		"PERF_COUNT_HW_INSTRUCTIONS",
		"MEM_UOPS_RETIRED.ALL_LOADS",
		"MEM_LOAD_UOPS_RETIRED.L1_MISS",
		"MEM_LOAD_UOPS_RETIRED.L2_MISS",
		"MEM_LOAD_UOPS_RETIRED.L3_MISS",
};

//static const char* DefaultPMU[] = {
//		"PERF_COUNT_HW_INSTRUCTIONS",
//		"MEM_UOPS_RETIRED.ALL_LOADS",
//		"MEM_LOAD_UOPS_RETIRED.L1_HIT",
//		"MEM_LOAD_UOPS_RETIRED.HIT_LFB",
//		"MEM_LOAD_UOPS_RETIRED.L1_MISS",
//};

static const char* L3HITPMU[] = {
		"MEM_UOPS_RETIRED.ALL_STORES",
		"MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_MISS",
		"MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_HIT",
		"MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_HITM",
};

static const char* L3MISSPMU[] = {
		"MEM_LOAD_UOPS_L3_MISS_RETIRED.LOCAL_DRAM",
		"MEM_LOAD_UOPS_L3_MISS_RETIRED.REMOTE_DRAM",
		"MEM_LOAD_UOPS_L3_MISS_RETIRED.REMOTE_HITM",
		"MEM_LOAD_UOPS_L3_MISS_RETIRED.REMOTE_FWD"
};

static const char* L2RQSTS[] = {
		"LONGEST_LAT_CACHE.MISS",
		"LONGEST_LAT_CACHE.REFERENCE",
		"L2_RQSTS.REFERENCES",
		"L2_RQSTS.MISS"
};

static const char* L2RQSTS2[] = {
		"L2_RQSTS.ALL_DEMAND_DATA_RD",
		"L2_RQSTS.ALL_CODE_RD",
		"L2_RQSTS.ALL_RFO",
		"MEM_LOAD_UOPS_RETIRED.L2_MISS",
};

static const char* L3AUX[] = {
		"MEM_LOAD_UOPS_RETIRED.L2_MISS",
		"MEM_LOAD_UOPS_RETIRED.L3_HIT",
		"MEM_LOAD_UOPS_RETIRED.L3_MISS",
		"PERF_COUNT_HW_CPU_CYCLES",
};

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))


static const char** select_pmu(const char* select_str, int& size){
	const char** pmu = nullptr;
	if (!strcmp(select_str, "default")){
		pmu = DefaultPMU;
		size = ARRAY_SIZE(DefaultPMU);
	}
	else if (!strcmp(select_str, "l3hit")){
		pmu = L3HITPMU;
		size = ARRAY_SIZE(L3HITPMU);
	}
	else if (!strcmp(select_str, "l3miss")){
		pmu = L3MISSPMU;
		size = ARRAY_SIZE(L3MISSPMU);
	}
	else if (!strcmp(select_str, "l2rqsts")){
		pmu = L2RQSTS;
		size = ARRAY_SIZE(L2RQSTS);
	}
	else if (!strcmp(select_str, "l2rqsts2")){
		pmu = L2RQSTS2;
		size = ARRAY_SIZE(L2RQSTS2);
	}
	else if (!strcmp(select_str, "l3aux")){
		pmu = L3AUX;
		size = ARRAY_SIZE(L3AUX);
	}
	else{
		size = 0;
	}
	return pmu;
}

class CacheStatMonitor{
public:
	// Counter Handler
	class CacheStatCounter{
		friend class CacheStateMonitor;
	public:
		// FIXME: This should be private
		CacheStatCounter(CacheStatMonitor& mon)
				: _mon(mon), _basestat{}
		{}
		void begin(){
			bool prev_state = _mon._enabled;
			_mon.stop();
			_basestat = _mon.read_counter();
			if (prev_state)
				_mon.start(false);
		}
		CacheStat end(){
			bool prev_state = _mon._enabled;
			_mon.stop();
			auto cur_count = _mon.read_counter();
			auto diff = cur_count - _basestat;
			_basestat = cur_count;
			if (prev_state)
				_mon.start(false);
			return diff;
		}
	private:
		CacheStatMonitor& _mon;
		CacheStat _basestat;
	};
public:
	// Methods
	void start(bool clear);
	void stop();
	void clear();
	CacheStatCounter *get_counter() { return new CacheStatCounter {*this}; }
	CacheStat read_counter();
	CacheStatMonitor(bool inherit, const char* pmu[], int size);
	~CacheStatMonitor();
private:
	// struct read_format{
	//     uint64_t nr;
	//     struct {
	//         uint64_t value;
	//         uint64_t id;
	//     } values [SZ_PMU];
	// };

	struct read_format{
		uint64_t value;
		uint64_t time_enabled;
		uint64_t time_running;
		uint64_t id;
	};
	const int num_pmu;
	perf_event_attr* _pe;
	int* _fd;
	int _enabled;
	bool _; // Dummy value
};


#endif //MODEL_CACHESTAT_MONITOR_HPP
