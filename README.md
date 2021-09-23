# NLP-Fast

This repository is the software artifacts for the paper entitled 'NLP-Fast: A Fast, Scalable, and Flexible System to Accelerate Large-Scale Heterogeneous NLP Models' published at PACT 2021.

The artifacts consist of three parts: (1) `CPU model` (CPU-version BERT implementation), (2) `GPU model` (GPU-version BERT implementation), and (3) `analysis tool` (various run scripts and analysis tools).
Please refer to the detailed instructions below.

## CPU model

We first introduce our CPU-version BERT implementation.
For the baseline implementation, we faithfully follow the TensorFlow's BERT model to build a representative baseline.
Then, we apply our `holistic model partitioning` on top of the baseline.
As model partitioning has three steps (i.e., partial-head, column-based, partial-ffw), we provide three types of implementations.
 1. partial_head
 2. column (partial-head + column-based)
 3. all_opt (partial-head + column-based + partial-ffw)

Also, to analyze the performance impacts of the proposed model partitioning optimizations, we provide two variants (i.e., clflush-version and prefetch-version).
To explicitly measure the performance impacts of the LLC cache misses, we put clflush instructions before every key operation.
To measure the maximum performance improvement, we explicitly loads the required before every operation.
For each variant, it has different suffix in the name of executable files. For instance,
 1. Original (suffix: .none)
 2. Explicit cache flush (suffix: .clflush)
 3. Explicit prefetching (suffix: .prefetch)

### How to build
Required packages => cmake, make, perf tools, Intel MKL

```
$cd {REPO_HOME}/model
$mkdir build
$cd build
$INTEL_MKL={INTEL_MKL_PATH};PERFMON={LIBPFM_PATH} cmake ..
$make -j
```

Then you can get all executables

### How to run?

You can get more detailed information by using a help option (-h).
A simple example is as follows. (By the way, we recommend users use the providied run scripts for their convenience. `./scripts/run_all.sh`)
Also, Please use our parser `./scripts/parse_all.py` to parse the results from run_all.sh script.
It will give you the csv file containing all profiled results as a dataframe (pandas).

```
./baseline.none -t 1 -b 1 -m 0 30522 2 8 16 256 1024 4096 24 -c64 (Not recommended)
```

### How to verify?

To verify our model implementation, we use the real weights and intermediate values that are extracted during SQuAD inference operations.
We use TensorFlow to dump those weights and intermediate values.
You can extract your own values or simply download the values we used. ([verification.tar.gz](https://github.com/SNU-HPCS/_backstore_/raw/main/verification.tar.gz))

```
$cd {$REPO_HOME}
$tar -zxvf verification.tar.gz
```

Then, you can run the verification script `./scripts/verify_all.sh`.
The results will be 

```
$./scripts/verify_all.sh
Start to verify baseline.none_t1_b1
difference, 2.06506e-06
Start to verify baseline.none_t1_b2
difference, 2.06506e-06
Start to verify baseline.none_t1_b4
difference, 2.06506e-06
Start to verify baseline.none_t1_b8
difference, 2.06506e-06
...

```

If the differnce is around 2.0e-06, it means the model is correct.


## GPU model

We also provide GPU-version BERT implementation.
For the baseline implementation, we faithfully follow the TensorFlow's GPU-version BERT model.
Then, we apply our `holistic model partitioning` on top of the baseline.
By applying the model partitinoing, we can use the CUDA stream to hide the memory accessing overhead (by overlapping memory operations with compute parts).
By doing so, we can improve the `single-GPU` performance.

Also, our model partitioning can improve the multi-GPU performance.
As our model partitioning minimizes the synchronization points, it gives significant performance improvement in the multi-GPU environment.
With faster interconnect (e.g., NVlink), we can achieve more scalable performance improvement.

There are three types of executable files:
 1. gpu_baseline_nostream: *baseline* without CUDA streams
 2. gpu_baseline: applying model partitioning optimizations and exploiting CUDA streams (single-GPU)
 3. gpu_multi_baseline: multi-GPU environment
 

### How to build
Required packages => cmake, make, Intel MKL, CUDA

```
$cd {REPO_HOME}/gpu_model
$mkdir build
$cd build
$INTEL_MKL={INTEL_MKL_PATH};CUDA_PATH={CUDA_PATH} cmake ..
$make -j
```

Then you can get all executables

### How to run?

Similar to the CPU case, we recommend the users use the providied run scripts (`./scripts/run_gpu_all.sh`).
Also, please use the pre-implemented parser `./scripts/parse_gpu_all.py` for the convenience.
The results will be dumped as a csv file including all results as a pandas dataframe.
Please, check help messages for more details.

### How to verify?

To verify the baseline implementation

```
$cd {REPO_HOME}/gpu_model/build
$./gpu_baseline_nostream -m 1 ../../verification/smallset ../../verification/weight
```

To verify the single-GPU implementation

```
$cd {REPO_HOME}/gpu_model/build
$./gpu_baseline -m 1 -s {num_streams} ../../verification/smallset ../../verification/weight
```

To verify the multi-GPU implementation

```
$cd {REPO_HOME}/gpu_model/build
$./gpu_multi_baseline -m 1 -n {num_gpus} ../../verification/smallset ../../verification/weight
```


## Analysis tool

All runnable scripts, parsers, and example analysis tools are located in `./scripts`.


## Troubleshoot

 1. verification mode doesn't work!
    * dump weights and intermediate data from tensorflow (while executing any workloads)
    * If you cannot understand the formats of verification files, please send an email at joonsung90@snu.ac.kr for more details

# Publication

This work has been published in [ISCA '19](https://dl.acm.org/doi/10.1145/3307650.3322214) and [PACT '21](TBA) ([Main talk](https://youtu.be/LOuqoVIage0))
