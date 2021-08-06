cmake_minimum_required(VERSION 3.10)
project(model)

set(CMAKE_CXX_STANDARD 14)

if(DEFINED ENV{INTEL_MKL})
    message(STATUS "Use Intel MKL library at $ENV{INTEL_MKL}")
else()
    message(FATAL_ERROR "Please set the environmental variable <INTEL_MKL>")
    exit()
endif()

if(DEFINED ENV{PERFMON})
    message(STATUS "Use Perf Monitor library at $ENV{PERFMON}")
else()
    message(FATAL_ERROR "Please set the environmental variable <PERFMON>")
    exit()
endif()

include_directories($ENV{INTEL_MKL}/mkl/latest/include)
include_directories($ENV{PERFMON}/include)

link_directories($ENV{INTEL_MKL}/mkl/latest/lib/intel64)
set(MKL_LIBS libmkl_intel_lp64.a libmkl_gnu_thread.a libmkl_core.a gomp)
link_directories($ENV{PERFMON}/lib)

set(MODEL_SRCS bert_main.cpp embedding.cpp
        tensor_func.cpp log.cpp utils.cpp
        bert_state.cpp cachestat.cpp cachestat_monitor.cpp)

#######################
##### 1. baseline
#######################
### 1.1. baseline
set(BASELINE_SRCS ${MODEL_SRCS} baseline_model.cpp)
set(BASELINE_LIBS ${MKL_LIBS} dl pthread pfm)
add_executable(baseline.none ${BASELINE_SRCS})
target_link_libraries(baseline.none ${BASELINE_LIBS})
target_compile_definitions(baseline.none PUBLIC -DBASELINE)

### 1.2. clflush-enabled baseline
add_executable(baseline.clflush ${BASELINE_SRCS})
target_link_libraries(baseline.clflush ${BASELINE_LIBS})
target_compile_definitions(baseline.clflush PUBLIC -DBASELINE -DCLFLUSH)

### 1.3. prefetch-enabled baseline
add_executable(baseline.prefetch ${BASELINE_SRCS})
target_link_libraries(baseline.prefetch ${BASELINE_LIBS})
target_compile_definitions(baseline.prefetch PUBLIC -DBASELINE -DPREFETCH)


#######################
##### 2. partial_head
#######################
### 2.1. partial_head
set(PARTIAL_HEAD_SRCS ${MODEL_SRCS} partial_head_model.cpp)
set(PARTIAL_HEAD_LIBS ${MKL_LIBS} dl pthread pfm)
add_executable(partial_head.none ${PARTIAL_HEAD_SRCS})
target_link_libraries(partial_head.none ${PARTIAL_HEAD_LIBS})
target_compile_definitions(partial_head.none PUBLIC -DPARTIAL_HEAD)

### 2.2. clflush-enabled partial_head
add_executable(partial_head.clflush ${PARTIAL_HEAD_SRCS})
target_link_libraries(partial_head.clflush ${PARTIAL_HEAD_LIBS})
target_compile_definitions(partial_head.clflush PUBLIC -DPARTIAL_HEAD -DCLFLUSH)

### 2.3. prefetch-enabled partial_head
add_executable(partial_head.prefetch ${PARTIAL_HEAD_SRCS})
target_link_libraries(partial_head.prefetch ${PARTIAL_HEAD_LIBS})
target_compile_definitions(partial_head.prefetch PUBLIC -DPARTIAL_HEAD -DPREFETCH)


#######################
##### 3. column (partial-head + column-based)
#######################
### 3.1. column
set(COLUMN_SRCS ${MODEL_SRCS} column_model.cpp)
set(COLUMN_LIBS ${MKL_LIBS} dl pthread pfm)
add_executable(column.none ${COLUMN_SRCS})
target_link_libraries(column.none ${COLUMN_LIBS})
target_compile_definitions(column.none PUBLIC -DCOLUMN)

### 3.2. clflush-enabled column
add_executable(column.clflush ${COLUMN_SRCS})
target_link_libraries(column.clflush ${COLUMN_LIBS})
target_compile_definitions(column.clflush PUBLIC -DCOLUMN -DCLFLUSH)

### 3.3. prefetch-enabled column
add_executable(column.prefetch ${COLUMN_SRCS})
target_link_libraries(column.prefetch ${COLUMN_LIBS})
target_compile_definitions(column.prefetch PUBLIC -DCOLUMN -DPREFETCH)


#######################
##### 4. all_opt (partial-head + column-based + partial-ffw) <Note: attention FC is missing>
#######################
### 4.1. all_opt
set(ALL_OPT_SRCS ${MODEL_SRCS} all_opt_model.cpp)
set(ALL_OPT_LIBS ${MKL_LIBS} dl pthread pfm)
add_executable(all_opt.none ${ALL_OPT_SRCS})
target_link_libraries(all_opt.none ${ALL_OPT_LIBS})
target_compile_definitions(all_opt.none PUBLIC -DALL_OPT)

### 4.2. clflush-enabled all_opt
add_executable(all_opt.clflush ${ALL_OPT_SRCS})
target_link_libraries(all_opt.clflush ${ALL_OPT_LIBS})
target_compile_definitions(all_opt.clflush PUBLIC -DALL_OPT -DCLFLUSH)

### 4.3. prefetch-enabled all_opt
add_executable(all_opt.prefetch ${ALL_OPT_SRCS})
target_link_libraries(all_opt.prefetch ${ALL_OPT_LIBS})
target_compile_definitions(all_opt.prefetch PUBLIC -DALL_OPT -DPREFETCH)


#######################
##### Others (ETC)
#######################
### softmax_perf_test
#add_executable(softmax_perf_test softmax_perf_test.cpp log.cpp cachestat.cpp cachestat_monitor.cpp)
#target_link_libraries(softmax_perf_test ${MKL_LIBS} dl pfm pthread)

### random chunk generator
#add_executable(random_generator random_chunk_generator.cpp)
#target_link_libraries(random_generator ${MKL_LIBS} dl pthread)

### Cache monitor test
add_executable(cache_monitor_test cache_monitor_test.cpp cachestat.cpp cachestat_monitor.cpp)
#set_target_properties(cache_monitor_test PROPERTIES COMPILE_FLAGS -O3)
target_link_libraries(cache_monitor_test ${MKL_LIBS} dl pthread pfm)