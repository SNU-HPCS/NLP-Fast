#ifndef GPU_MODEL_LOG_HPP
#define GPU_MODEL_LOG_HPP
#include <chrono>

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
#endif //GPU_MODEL_LOG_HPP