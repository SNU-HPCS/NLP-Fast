#ifndef MODEL_CACHESTAT_HPP
#define MODEL_CACHESTAT_HPP
#include <cstdint>
#include <cstring>
#include <cstdio>
#include "enum_util.hpp"

static uint64_t wraparound(uint64_t a, uint64_t b){
	if (a >= b)
		return a - b;
	else{
		printf("[Warn] wraparound happen %lu, %lu %lu\n", a, b, (((uint64_t) -1) - b) + a);
		return (((uint64_t) -1) - b) + a;
	}
}

#define _ENUM_CS_DATA(E)                        \
    E(CS_INST)                                  \
    E(CS_LD)                                    \
    E(CS_L1MISS)                                \
    E(CS_L2MISS)                                \
    E(CS_L3MISS)                                \

//#define _ENUM_CS_DATA(E)                        \
//    E(CS_INST)                                  \
//    E(CS_LD)                                    \
//    E(CS_L1HIT)                                 \
//    E(CS_L2HIT)                                 \
//    E(CS_L3HIT)                                 \

struct CacheStat{
	// This should be listed in the same order of PMU.
	enum ENUM_CS_DATA{
		_ENUM_CS_DATA(GEN_ENUM)
		NUM_CS_DATA
	};
	static const char* STR_ENUM_CS_DATA[NUM_CS_DATA];
	struct Value{
		uint64_t value;
		uint64_t time_enabled;
		uint64_t time_running;
	};
	Value data [NUM_CS_DATA] __attribute__ ((aligned(64)));

	CacheStat(){
		memset(data, 0, sizeof(data));
	}
	CacheStat& operator+=(const CacheStat& rhs){
		for (int i = 0; i < NUM_CS_DATA; ++i){
			data[i].value += rhs.data[i].value;
			data[i].time_enabled += rhs.data[i].time_enabled;
			data[i].time_running += rhs.data[i].time_running;
		}
		return *this;
	}
	CacheStat& operator-=(const CacheStat& rhs){
		for (int i = 0; i < NUM_CS_DATA; ++i){
			// Wraparound can happen
			data[i].value = wraparound(data[i].value, rhs.data[i].value);
			data[i].time_enabled =
					wraparound(data[i].time_enabled, rhs.data[i].time_enabled);
			data[i].time_running = wraparound(data[i].time_running,
			                                  rhs.data[i].time_running);
		}
		return *this;
	}
};

CacheStat operator+(const CacheStat& lhs, const CacheStat& rhs);
CacheStat operator-(const CacheStat& lhs, const CacheStat& rhs);
void print_cs(const CacheStat& cs);

#endif //MODEL_CACHESTAT_HPP
