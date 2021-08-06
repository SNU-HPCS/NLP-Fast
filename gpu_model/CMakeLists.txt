cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CUDA_ARCHITECTURES 50)    # Need to be specified if cuda kernel call returns zero values (forcing to use the proper CUDA version)

if(DEFINED ENV{INTEL_MKL})
    message(STATUS "Use Intel MKL library at $ENV{INTEL_MKL}")
else()
    message(FATAL_ERROR "Please set the environmental variable <INTEL_MKL>")
    exit()
endif()

if(DEFINED ENV{CUDA_PATH})
    message(STATUS "Use CUDA library at $ENV{CUDA_PATH}")
else()
    message(FATAL_ERROR "Please set the environmental variable <CUDA_PATH>")
    exit()
endif()
set(CMAKE_CUDA_COMPILER $ENV{CUDA_PATH}/bin/nvcc)

project(gpu_model)
enable_language(CUDA)

include_directories($ENV{INTEL_MKL}/mkl/latest/include)
include_directories($ENV{CUDA_PATH}/include)

link_directories($ENV{INTEL_MKL}/mkl/latest/lib/intel64)

#set(MKL_LIBS libmkl_intel_lp64.a libmkl_gnu_thread.a libmkl_core.a gomp)
set(MKL_LIBS mkl_intel_lp64 mkl_gnu_thread mkl_core gomp)
set(GPU_MODEL_CPU_SRCS gpu_bert_main.cpp bert_op_cpu.cpp
        bert_state.cpp log.cpp utils.cpp tensor_op_cpu.cpp)
set(GPU_MODEL_GPU_SRCS
        cuda_init.cu cuda_init.cuh
        tensor_op_gpu.cu tensor_op_gpu.cuh)

add_executable(gpu_baseline
        ${GPU_MODEL_CPU_SRCS}
        ${GPU_MODEL_GPU_SRCS})
set_target_properties(gpu_baseline
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_baseline
        PROPERTIES
        COMPILE_FLAGS -std=c++14)
target_link_libraries(gpu_baseline ${MKL_LIBS} cublas cusparse pthread cudart)


set(GPU_MODEL_NOSTREAM_CPU_SRCS gpu_bert_main.cpp bert_op_cpu.cpp
        bert_state.cpp log.cpp utils.cpp tensor_op_cpu.cpp)
set(GPU_MODEL_NOSTREAM_GPU_SRCS
        cuda_nostream_init.cu cuda_nostream_init.cuh
        tensor_op_gpu.cu tensor_op_gpu.cuh)

add_executable(gpu_baseline_nostream
        ${GPU_MODEL_NOSTREAM_CPU_SRCS}
        ${GPU_MODEL_NOSTREAM_GPU_SRCS})
set_target_properties(gpu_baseline_nostream
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_baseline_nostream
        PROPERTIES
        COMPILE_FLAGS -std=c++14)
target_link_libraries(gpu_baseline_nostream ${MKL_LIBS} cublas cusparse pthread cudart)


set(GPU_MULTI_MODEL_CPU_SRCS gpu_multi_bert_main.cpp bert_op_cpu.cpp
        bert_state.cpp log.cpp utils.cpp tensor_op_cpu.cpp)
set(GPU_MULTI_MODEL_GPU_SRCS
        cuda_multi_init.cu cuda_multi_init.cuh
        tensor_op_gpu.cu tensor_op_gpu.cuh)

add_executable(gpu_multi_baseline
        ${GPU_MULTI_MODEL_CPU_SRCS}
        ${GPU_MULTI_MODEL_GPU_SRCS})
set_target_properties(gpu_multi_baseline
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_multi_baseline
        PROPERTIES
        COMPILE_FLAGS -std=c++14)
target_link_libraries(gpu_multi_baseline ${MKL_LIBS} cublas cusparse pthread cudart)
