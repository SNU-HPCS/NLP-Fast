#include <cstdio>
#include "log.hpp"

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