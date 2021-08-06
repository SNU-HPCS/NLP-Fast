#include "cachestat_monitor.hpp"
#include <linux/hw_breakpoint.h>
#include <linux/perf_event.h>
#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_perf_event.h>
#include <stdlib.h>
#include <cassert>

CacheStatMonitor::CacheStatMonitor(bool inherit,
		const char* pmu[],
		int num_pmu) : num_pmu(num_pmu), _pe (nullptr), _fd (nullptr), _enabled (false)
{
	static bool _init = pfm_initialize();
	_ = _init;
	assert (num_pmu <= 5);
	_pe = new perf_event_attr [num_pmu];
	assert(_pe);
	_fd = new int [num_pmu];
	assert(_fd);
	memset(_fd, 0, sizeof(_fd[0]) * num_pmu);
	memset(_pe, 0, sizeof(_pe[0]) * num_pmu);
	// FIXME: currently it just uses a single group.
	// only possible to measure 4 events (+ fixed counter event)
	// at the same time.
//	printf("count %d\n", num_pmu);
	for (int i = 0; i < num_pmu; ++i){
//		printf("set %s\n", pmu[i]);
		int err = pfm_get_perf_event_encoding(pmu[i], PFM_PLM3 | PFM_PLM0,
		                                      &_pe[i], NULL, NULL);
		if (err != PFM_SUCCESS){
			printf("Error: %d", err);
			exit(err);
		}
		//printf("Setup PMU:%s, type:%x, config:%llx, inherit:%d\n", PMU[i],
		//_pe[i].type, _pe[i].config, (int)inherit);
		_pe[i].size = sizeof(struct perf_event_attr);
		_pe[i].read_format = PERF_FORMAT_ID | PERF_FORMAT_ID |
		                     PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
		_pe[i].disabled = 1;
		_pe[i].inherit = (int) inherit;
		_pe[i].exclude_kernel = 1;
		_pe[i].exclude_hv = 1;

		int group_fd = (i ? _fd[0] : -1);
		int fd = perf_event_open(&_pe[i], 0, -1, group_fd, 0);
		if (fd == -1){
			fprintf(stderr, "Error opening leader %llx\n", _pe[i].config);
			exit(EXIT_FAILURE);
		}
		_fd[i] = fd;
	}
	ioctl(_fd[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
}

// Enable cache stat counting
// Have to clear counter explicitly if it is required.
void CacheStatMonitor::start(bool clear){
	if (clear)
		ioctl(_fd[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
	if (!_enabled){
		ioctl(_fd[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
		_enabled = true;
	}
}

// Diable cache stat counting
// Have to clear counter explicitly if it is required.
void CacheStatMonitor::stop(){
	if (_enabled){
		ioctl(_fd[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
		_enabled = false;
	}
}

// Clear all cache stat PMUs
void CacheStatMonitor::clear(){
	ioctl(_fd[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
}

// Read cache stat PMUs
CacheStat CacheStatMonitor::read_counter(){
	CacheStat res;
	read_format count;
	memset(&count, 0, sizeof(count));
	for (int i = 0; i < num_pmu; ++i){
		int err = read(_fd[i], &count, sizeof(count));
		if (err == -1){
			fprintf(stderr, "Error while reading counter\n");
			exit(1);
		}
		res.data[i].value = count.value;
		res.data[i].time_enabled = count.time_enabled;
		res.data[i].time_running = count.time_running;
	}
	return res;
}

CacheStatMonitor::~CacheStatMonitor(){
	if (_fd){
		for (int i = 0; i < num_pmu; ++i)
			close(_fd[i]);
		delete [] _fd;
	}
	if (_pe)
		delete [] _pe;
}

#ifdef DEBUG

void do_something(){
    const int bufsize = 100000;
    char* buf = (char*) malloc(bufsize);
    for (int i = 0; i < bufsize; ++i){
        buf[i] = i;
    }
    free((void*)buf);
}

int main(){
    CacheStatMonitor csmon;
    CacheStat cs;
    print_cs(cs);
    auto cscounter = csmon.get_counter();
    auto cscounter2 = csmon.get_counter();
    cscounter.begin();
    cscounter2.begin();
    csmon.start(false);
    do_something();
    cs += cscounter.end();
    print_cs(cs);
    cscounter.begin();
    do_something();
    cs += cscounter.end();
    print_cs(cs);
    printf("Total\n");
    print_cs(cscounter2.end());
    return 0;
}
#endif
