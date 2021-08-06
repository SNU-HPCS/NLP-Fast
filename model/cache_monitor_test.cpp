#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mkl.h>
#include <sys/resource.h>

#include "log.hpp"

/// Random access
static inline int next(size_t *array, unsigned int sz_array, unsigned int i) {
	return (22695477u * i + 1) &
	       (sz_array/sizeof(size_t) - 1);
}

static void init_array(size_t *array, unsigned int sz_array) {
	unsigned int i = 0;

	do {
		int next_i = next(array, sz_array, i);
		array[i] = (size_t)&array[next_i];
		i = next_i;
	} while (i);
}
static size_t *arr_base;
static void access_array_rand(size_t *array, unsigned int sz_array) {
	volatile size_t *ptr = array;

	for (int i = 0; i < sz_array; i++) {
		//printf("ptr (%p) (offset) => %d\n",ptr, ptr - arr_base);
		ptr = (volatile size_t *)*ptr;
	}
}

/// Sequential access
static void access_array_seq(size_t *array, unsigned int sz_array) {
	volatile size_t *ptr = array;
	volatile size_t tmp;

	for (int i = 0; i < sz_array; i++) {
		tmp = ptr[i];
		asm volatile("mfence":::"memory");
	}
}

static void flush_cache(void* data, unsigned long size) {
	char *c_data = (char*)data;
	const int CL_SIZE = 32;
	for (long i = 0; i < size; i += CL_SIZE){
		char* p = c_data + i;
		asm volatile ("clflush %0" :: "m" (*(char*) p) : "memory");
	}
	asm volatile("mfence":::"memory");

//	// tlb flush
//	if (!ioctlfd) {
//		ioctlfd = open("/dev/flush_tlb", O_RDWR);
//		if (ioctlfd < 0) {
//			perror("Open flush_tlb binder failed");
//			exit(1);
//		}
//	}
//
//	unsigned long addr = 0;
//	if (ioctl(ioctlfd, FLUSH_TLB_CMD, &addr)) {
//		perror("why");
//		exit(1);
//	}
}

static void prefetch_data(void *data, unsigned long size) {
	volatile char c_tmp = 0;
	char *c_data = (char*)data;

	for (long i = size-1; i>=0; i--) {
		c_tmp += c_data[i];
		asm volatile("mfence":::"memory");
	}
	asm volatile("mfence":::"memory");
}

int main(int argc, char *argv[]) {
	size_t *buffer;
	int size;
	int _pmu_size;
	const char **pmu = select_pmu("default", _pmu_size); // Todo: get pmu_type from a command argument
	CacheStatMonitor *csmon = new CacheStatMonitor(false, pmu, _pmu_size);
	CacheStat cs_start, cs_end, cs_diff;
	LatCounter lat_counter;
	LatCounter lat_counter2;

	if (argc != 3) {
		printf("Usage %s [size:KB] [mode: 0 (seq), 1 (rand)\n", argv[0]);
		exit(1);
	}

	int which = PRIO_PROCESS;
	id_t pid;
	int priority = -20;
	int ret;

	pid = getpid();
	ret = setpriority(which, pid, priority);


	size = 1024 * atoi(argv[1]);
	//size_t test_access_count = size / sizeof(size_t) / 16;
	size_t test_access_count = size / sizeof(size_t);
	printf("access_count: %ld\n",test_access_count);
	buffer = (size_t*)mkl_malloc(size, 64);
	arr_base = buffer;
	init_array(buffer, size);
	int mode = atoi(argv[2]);


	/// Test!!!! (prefetch)
	//prefetch_data(buffer, size);

	//csmon->stop();
	//cs_start = csmon->read_counter();
	//csmon->start(false);
	//lat_counter.begin();

	//if (mode == 0) {
		//access_array_seq(buffer, test_access_count);
	//} else if (mode == 1) {
		//access_array_rand(buffer, test_access_count);
	//} else {
		//assert(0);
	//}

	//lat_counter.end();
	//csmon->stop();
	//cs_end = csmon->read_counter();
	//cs_diff = cs_end - cs_start;
	//printf("[prefetch] latency: %lf\n",lat_counter.get_latency());
	//for (int i = 0; i < 5; i++) {
		//printf("    %10s %lu\n", cs_diff.STR_ENUM_CS_DATA[i], cs_diff.data[i].value);
	//}

	/// Test!!!! (clflush)
	prefetch_data(buffer, size);
	flush_cache(buffer, size);
	lat_counter.begin();
	lat_counter.end();

	csmon->stop();
	cs_start = csmon->read_counter();
	csmon->start(false);
	lat_counter.begin();

	if (mode == 0) {
		access_array_seq(buffer, test_access_count);
	} else if (mode == 1) {
		access_array_rand(buffer, test_access_count);
	} else {
		assert(0);
	}

	lat_counter.end();
	csmon->stop();
	cs_end = csmon->read_counter();
	cs_diff = cs_end - cs_start;
	printf("[flush] latency: %lf\n",lat_counter.get_latency());
	for (int i = 0; i < 5; i++) {
		printf("    %10s %lu\n", cs_diff.STR_ENUM_CS_DATA[i], cs_diff.data[i].value);
	}

	/// Test!!!! (prefetch)
	prefetch_data(buffer, size);

	csmon->stop();
	cs_start = csmon->read_counter();
	csmon->start(false);
	lat_counter.begin();

	if (mode == 0) {
		access_array_seq(buffer, test_access_count);
	} else if (mode == 1) {
		access_array_rand(buffer, test_access_count);
	} else {
		assert(0);
	}

	lat_counter.end();
	csmon->stop();
	cs_end = csmon->read_counter();
	cs_diff = cs_end - cs_start;
	printf("[prefetch] latency: %lf\n",lat_counter.get_latency());
	for (int i = 0; i < 5; i++) {
		printf("    %10s %lu\n", cs_diff.STR_ENUM_CS_DATA[i], cs_diff.data[i].value);
	}

	/// Test!!!! (clflush)
	flush_cache(buffer, size);

	csmon->stop();
	cs_start = csmon->read_counter();
	csmon->start(false);
	lat_counter.begin();

	if (mode == 0) {
		access_array_seq(buffer, test_access_count);
	} else if (mode == 1) {
		access_array_rand(buffer, test_access_count);
	} else {
		assert(0);
	}

	lat_counter.end();
	csmon->stop();
	cs_end = csmon->read_counter();
	cs_diff = cs_end - cs_start;
	printf("[flush] latency: %lf\n",lat_counter.get_latency());
	for (int i = 0; i < 5; i++) {
		printf("    %10s %lu\n", cs_diff.STR_ENUM_CS_DATA[i], cs_diff.data[i].value);
	}
}
