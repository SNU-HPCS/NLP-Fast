#!/bin/bash
BASEDIR=$(dirname $0)
source ${BASEDIR}/eval_type.conf

threadblocks=(
    "32"
    "128"
    "512"
)

num_batchs=(
    "1"
    "2"
    "4"
    "8"
)

# [# of batch threads]:[# of head threads]
# (NOTE) # of head threads is used for library threads
GPUID_NUMSTREAMS=(
    "0:1"
    "1:2"
    "2:4"
    "3:8"
    "4:16"
)

MULTI_GPUN_NUMS=(
    "1"
    "2"
    "4"
)

if [ "$#" -eq "2" ]; then
    TESTCASE=$1
    ITER_NUM=$2
else
    echo "Usage $0 [testcase] [iter_num]"
    exit 1
fi

DATE=$(date "+%y%m%d%H%M%S")
TESTNAME="gpu_${TESTCASE}_${DATE}"
BINDIR=${BASEDIR}/../gpu_model/build
LOGDIR=${BASEDIR}/../expdata/${TESTNAME}
RANDOM_CHUNK_DIR=${BASEDIR}/../random_chunks

########################################
############## TEST function
########################################
function start_all {
    bin_path=${1}
    bin_multi_gpu_path=${2}
    seq_length=${3}
    hidden_size=${4}
    num_head=${5}
    ff_size=${6}
    num_batch=${7}
    threadblock=${8}
    iter_num=${9}

    ### Single-GPU multi-stream
    args="-m 0 ${vocab_sz} ${token_sz} ${num_batch} ${num_head} ${seq_length} ${hidden_size} ${ff_size} ${num_layer} ${RANDOM_CHUNK_DIR} -g {1} -s {2} -b ${threadblock}"
    exec_log_dir="${LOGDIR}/single-gpu_stream-{2}_tb-${threadblock}_batch-${num_batch}_head-${num_head}_layer-${num_layer}_seq-${seq_length}_hsize-${hidden_size}_ffsize-${ff_size}_iter-${iter_num}"

    # Execution
    cmd="mkdir -p ${exec_log_dir};"
    cmd+="${bin_path} $args > ${exec_log_dir}/log.txt 2>&1"
    parallel --verbose -C: "$cmd" ::: "${GPUID_NUMSTREAMS[@]}"


    ### multi-GPU single-stream
    args="-m 0 ${vocab_sz} ${token_sz} ${num_batch} ${num_head} ${seq_length} ${hidden_size} ${ff_size} ${num_layer} ${RANDOM_CHUNK_DIR} -n {1} -b ${threadblock}"
    exec_log_dir="${LOGDIR}/multi-gpu_stream-{1}_tb-${threadblock}_batch-${num_batch}_head-${num_head}_layer-${num_layer}_seq-${seq_length}_hsize-${hidden_size}_ffsize-${ff_size}_iter-${iter_num}"

    # Execution
    cmd="mkdir -p ${exec_log_dir};"
    cmd+="${bin_multi_gpu_path} $args > ${exec_log_dir}/log.txt 2>&1"
    parallel --verbose -C: -j1 "$cmd" ::: "${MULTI_GPUN_NUMS[@]}"
}

########################################
############## START eval
########################################

eval_list=(
    #"bert_small"
    #"bert_small_h256"
    #"bert_base"
    "bert_large"
    "bert_large_seq128"
    "bert_large_seq512"
    #"bert_large_seq1024"
    #"bert_large_seq4096"
    "bert_large_h512"
    "bert_large_h2048"
    #"bert_large_h4096"
    #"bert_large_h16384"
    "bert_large_ff2048"
    "bert_large_ff8192"
    #"bert_large_ff16384"
    "megatron_hh1536"
    #"megatron_hh1920"
    #"megatron_hh2304"
    "megatron_hh3072"
    #"small_for_debug"
)

for eval_type in "${eval_list[@]}"; do
    seq_length=${seq_lengths[${eval_type}]}
    hidden_size=${hidden_sizes[${eval_type}]}
    num_head=${num_heads[${eval_type}]}
    ff_size=${ff_sizes[${eval_type}]}

    # DO Test!!!
    for threadblock in "${threadblocks[@]}"; do
        for num_batch in "${num_batchs[@]}"; do
            for iter in $(seq "1" "$ITER_NUM"); do
                start_all ${BINDIR}/gpu_baseline ${BINDIR}/gpu_multi_baseline ${seq_length} ${hidden_size} ${num_head} ${ff_size} ${num_batch} ${threadblock} ${iter}
            done
        done
    done
done

echo "=========================================="
echo "[DONE]"
echo "    TESTCASE:${TESTCASE}"
echo "    DATE:${DATE}"
echo "    TESTNAME:${TESTNAME}"
echo "    LOGDIR:${LOGDIR}"
echo "    eval_types:${eval_list[@]}"
echo "=========================================="
