#!/bin/bash
BASEDIR=$(dirname $0)
source ${BASEDIR}/eval_type.conf

# [# of batch threads]:[# of head threads]
# (NOTE) # of head threads is used for library threads
BATCH_HEAD_THREADS=(
    "1:1"
    "1:4"
	"1:16"
    "2:1"
    "2:4"
	"2:16"
    "4:1"
    "4:4"
    "4:8"
    "8:1"
    "8:4"
)

if [ "$#" -eq "1" ]; then
    TESTCASE=$1
    #RUN_MODES=("baseline" "partial_head" "column" "all_opt")
    #OPTION_POSTFIX=("none" "clflush" "prefetch")
    RUN_MODES=("baseline" "all_opt")
    OPTION_POSTFIX=("none" "clflush" "prefetch")
elif [ "$#" -ge "2" ]; then
    TESTCASE=$1
    RUN_MODES=("baseline" "partial_head" "column" "all_opt")
    if [ "$2" == "baseline" ]; then
        RUN_MODES=("baseline")
    elif [ "$2" == "partial_head" ]; then
        RUN_MODES=("partial_head")
    elif [ "$2" == "column" ]; then
        RUN_MODES=("column")
    elif [ "$2" == "all_opt" ]; then
        RUN_MODES=("all_opt")
    elif [ "$2" == "all" ]; then
        RUN_MODES=("baseline" "partial_head" "column" "all_opt")
    else
        echo "Invalid RUN_MODE ($2)"
        exit 1
    fi

    OPTION_POSTFIX=("none" "clflush" "prefetch")
    if [ "$#" -eq "3" ]; then
        if [ "$3" == "none" ]; then
            OPTION_POSTFIX=("none")
        elif [ "$3" == "clflush" ]; then
            OPTION_POSTFIX=("clflush")
        elif [ "$3" == "prefetch" ]; then
            OPTION_POSTFIX=("prefetch")
        elif [ "$3" == "all" ]; then
            OPTION_POSTFIX=("none" "clflush" "prefetch")
        else
            echo "Invalid OPTION ($3)"
            exit 1
        fi
    fi
else
    echo "Usage $0 [testcase]"\
        "[RUN_MODE: baseline/partial_head/column/all_opt/all, default: all]"\
        "[option: clflush/prefetch/none/all, default: all]"
    exit 1
fi

DATE=$(date "+%y%m%d%H%M%S")
TESTNAME=${TESTCASE}_${DATE}
BINDIR=${BASEDIR}/../model/build
LOGDIR=${BASEDIR}/../expdata/${TESTNAME}
RANDOM_CHUNK_DIR=${BASEDIR}/../random_chunks

########################################
############## TEST function
########################################
function start_all {
    run_mode=${1}
    option=${2}
    bin_path=${3}
    seq_length=${4}
    hidden_size=${5}
    num_head=${6}
    ff_size=${7}
    chunk_size=${8}

    args="-b {1} -t {2} -m 0 ${vocab_sz} ${token_sz} ${num_batch} ${num_head} ${seq_length} ${hidden_size} ${ff_size} ${num_layer} -c ${chunk_size} ${RANDOM_CHUNK_DIR}"
    exec_log_dir="${LOGDIR}/${run_mode}.${option}_bth-{1}_lth-{2}_batch-${num_batch}_head-${num_head}_layer-${num_layer}_seq-${seq_length}_hsize-${hidden_size}_ffsize-${ff_size}"

    # Execution
    cmd="mkdir -p ${exec_log_dir};"
    cmd+="${bin_path} $args > ${exec_log_dir}/log.txt 2>&1"
    parallel --verbose -C: -j1 "$cmd" ::: "${BATCH_HEAD_THREADS[@]}"
}

########################################
############## START eval
########################################

eval_list=(
    #"bert_small"
    #"bert_small_h256"
    #"bert_base"
    "bert_large"
    #"bert_large_seq128"
    ##"bert_large_seq512"
    "bert_large_seq1024"
    #"bert_large_seq4096"
    #"bert_large_h512"
    ##"bert_large_h2048"
    "bert_large_h4096"
    #"bert_large_h16384"
    ##"bert_large_ff2048"
    ##"bert_large_ff8192"
    "bert_large_ff16384"
    #"megatron_hh1536"
    #"megatron_hh1920"
    #"megatron_hh2304"
    #"megatron_hh3072"
    ##"small_for_debug"
)

for eval_type in "${eval_list[@]}"; do
    seq_length=${seq_lengths[${eval_type}]}
    hidden_size=${hidden_sizes[${eval_type}]}
    num_head=${num_heads[${eval_type}]}
    ff_size=${ff_sizes[${eval_type}]}
    chunk_size=64

    # Do TEST!!!
    for run_mode in "${RUN_MODES[@]}"; do
        for option in "${OPTION_POSTFIX[@]}"; do
            #echo "[DEBUG] run_mode: ${run_mode}, option: ${option}"
            start_all ${run_mode} ${option} ${BINDIR}/${run_mode}.${option} ${seq_length} ${hidden_size} ${num_head} ${ff_size} ${chunk_size}
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
