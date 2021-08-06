#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#include <limits.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "matmul.h"
//#include "thrust_stride.h"

// Helpers
#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
    fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
    exit(-1);}} while(0)
#define CUDA_CALL(X)     ERR_NE((X), cudaSuccess)
#define CUBLAS_CALL(X) ERR_NE((X), CUBLAS_STATUS_SUCCESS)
#define CUSPARSE_CALL(X) ERR_NE((X), CUSPARSE_STATUS_SUCCESS)

#define ZERO_SKIP 1

#define MAX(X, Y)       ((X > Y) ? X : Y)
#define CEILING(X, Y)   (((X) + (Y) - 1) / (Y))

// Apply exp element-wisely to an array d_A
// d_A(m) = exp(d_A(n))
__global__ void g_exp(float *A, const int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n)
        return;
    /*
    float x = __expf(A[i]);
    A[i] = ZERO_SKIP && x < 0.1 ? 0 : x;
    */
    A[i] = __expf(A[i]);
}

// Divide PQ by S for each question
// m: num_sentences
// n: num_questions
__global__ void g_normalize(float *PQ, const float *S, const int n, const int mn)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= mn)
        return;
    int y = i / n;
    int x = i % n;
    PQ[n * y + x] /= S[x];
}

// Main
int main(int argc, char *argv[]) {
    /*
     * EIM: NUM_SENTENCES x EMBED_DIM
     * EQ: EMBED_DIM x 1
     * P <= EIM[i] x EQ
     * EOM[i]: EMBED_DIM x NUM_SENTENCES
     * P: NUM_SENTENCES x 1
     * O <= EOM[i] x P
     */

    int gpu_id, block_size, num_streams;
    int num_sentences, embed_dim, num_questions;

    if (argc == 1) {
        gpu_id = 0;
        block_size = 32;
        num_streams = 1;
        num_sentences = 10000000;
        embed_dim = 50;
        num_questions = 1;
    } else if (argc == 7) {
        int i = 0;
        gpu_id = atoi(argv[++i]);
        block_size = atoi(argv[++i]);
        num_streams = atoi(argv[++i]);
        num_sentences = atoi(argv[++i]);
        embed_dim = atoi(argv[++i]);
        num_questions = atoi(argv[++i]);
    } else {
        printf("Usage: %s gpu_id block_size num_streams "
                "num_senteces embed_dim num_questions\n", argv[0]);
        return -1;
    }
    if (argc == 1)
        printf("(running with built-in args) ");
    printf("executable=%s "
            "gpu_id=%d block_size=%d num_streams=%d "
            "num_sentences=%d embed_dim=%d num_questions=%d\n",
            argv[0],
            gpu_id, block_size, num_streams,
            num_sentences, embed_dim, num_questions
          );
    fflush(stdout);

    assert(num_streams <= num_sentences);

    int num_gpus;
    CUDA_CALL( cudaGetDeviceCount(&num_gpus) );
    printf("deviceCount=%d\n", num_gpus);
    CUDA_CALL( cudaSetDevice(gpu_id) );

    cudaDeviceProp prop;
    CUDA_CALL( cudaGetDeviceProperties(&prop, 0) );
    printf("deviceOverlap=%d\n", prop.deviceOverlap);
    int hasHyperQ = (prop.major < 3 || (prop.major == 3 && prop.minor < 5)) ? 0 : 1;
    printf("hasHyperQ=%d\n", hasHyperQ);

    printf("main begin\n"); fflush(stdout);

    // Timestamps
    struct timeval tv_begin, tv_end;

    // Init CUDA streams, cublas contexts
    cudaStream_t streams[num_streams];
    cublasHandle_t cublas_handles[num_streams];
    cusparseHandle_t cusparse_handles[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        CUBLAS_CALL( cublasCreate(&cublas_handles[i]) );
        CUBLAS_CALL( cublasSetStream(cublas_handles[i], streams[i]) );
        CUSPARSE_CALL( cusparseCreate(&cusparse_handles[i]) );
        CUSPARSE_CALL( cusparseSetStream(cusparse_handles[i], streams[i]) );
    }

    size_t size;

    // Embedded input
    float *h_EIM, *d_EIM;
    size = sizeof(float) * num_sentences * embed_dim;
    CUDA_CALL( cudaMallocHost(&h_EIM, size) );
    CUDA_CALL( cudaMalloc(&d_EIM, size) );
    for (int i = 0; i < num_sentences; i++)
        for (int j = 0; j < embed_dim; j++)
            h_EIM[i * embed_dim + j] = 0.1;

    // Embedded output
    float *h_EOM, *d_EOM;
    size = sizeof(float) * num_sentences * embed_dim;
    CUDA_CALL( cudaMallocHost(&h_EOM, size) );
    CUDA_CALL( cudaMalloc(&d_EOM, size) );
    for (int i = 0; i < num_sentences; i++)
        for (int j = 0; j < embed_dim; j++)
            h_EOM[i * embed_dim + j] = 0.1;

    // Embedded question
    float *h_EQ, *d_EQ;
    size = sizeof(float) * embed_dim * num_questions;
    CUDA_CALL( cudaMallocHost(&h_EQ, size) );
    CUDA_CALL( cudaMalloc(&d_EQ, size) );
    for (int i = 0; i < embed_dim; i++)
        for (int j = 0; j < num_questions; j++)
            h_EOM[i * num_questions + j] = 0.1;
    CUDA_CALL( cudaMemcpy(d_EQ, h_EQ, size, cudaMemcpyHostToDevice) );

    // Probability
    float *d_PQ;
    size = sizeof(float) * num_sentences * num_questions;
    CUDA_CALL( cudaMalloc(&d_PQ, size) );

    // Compation mask
    float *d_MQ;
    size = sizeof(float) * num_sentences * num_questions;
    CUDA_CALL( cudaMalloc(&d_MQ, size) );

    // Stream probabilty sum
    float *d_SS;
    size = sizeof(float) * num_streams * num_questions;
    CUDA_CALL( cudaMalloc(&d_SS, size) );

    // Probabilty sum
    float *d_S;
    size = sizeof(float) * num_questions;
    CUDA_CALL( cudaMalloc(&d_S, size) );

    // Ones vector
    float *h_OS, *d_OS;
    size = sizeof(float) * num_sentences;
    CUDA_CALL( cudaMallocHost(&h_OS, size) );
    CUDA_CALL( cudaMalloc(&d_OS, size) );
    for (int i = 0; i < num_sentences; i++)
        h_OS[i] = 1;
    CUDA_CALL( cudaMemcpy(d_OS, h_OS, size, cudaMemcpyHostToDevice) );

    // Stream output
    float *d_SOM;
    size = sizeof(float) * num_streams * num_questions * embed_dim;
    CUDA_CALL( cudaMalloc(&d_SOM, size) );

    // Output
    float *h_OM, *d_OM;
    size = sizeof(float) * num_questions * embed_dim;
    CUDA_CALL( cudaMallocHost(&h_OM, size) );
    CUDA_CALL( cudaMalloc(&d_OM, size) );

    CUDA_CALL( cudaDeviceSynchronize() );
    gettimeofday(&tv_begin, NULL);

    // NOTE: for i == num_streams - 1, need to add num_sentences % num_streams
    int num_block_sentences_base = num_sentences / num_streams;

    // EIM to device
    for (int i = 0; i < num_streams; i++) {
        int num_block_sentences = i == num_streams - 1 ?
                num_block_sentences_base + num_sentences % num_streams :
                num_block_sentences_base;
        int size = sizeof(float) * num_block_sentences * embed_dim;;
        int offset = num_block_sentences_base * embed_dim * i;
        CUDA_CALL( cudaMemcpyAsync(d_EIM + offset, h_EIM + offset, size, cudaMemcpyHostToDevice, streams[i]) );
    }

    // Kernel; PQ = EIM * EQ
    for (int i = 0; i < num_streams; i++) {
        int num_block_sentences = i == num_streams - 1 ?
                num_block_sentences_base + num_sentences % num_streams :
                num_block_sentences_base;
        float alpha = 1., beta = 0.;
        CUBLAS_CALL(cublasSgemm(cublas_handles[i],
                CUBLAS_OP_N, CUBLAS_OP_N,
                num_block_sentences, num_questions, embed_dim,
                &alpha,
                d_EIM + num_block_sentences_base * embed_dim * i, num_block_sentences,
                d_EQ, embed_dim,
                &beta,
                d_PQ + num_block_sentences_base * num_questions * i, num_block_sentences
                ));
        /*
        matmul<1, 128, 16>(streams[i],
                num_block_sentences, embed_dim, num_questions,
                d_EIM + num_block_sentences_base * embed_dim * i,
                d_EQ,
                d_PQ + num_block_sentences_base * num_questions * i);
        CUDA_CALL( cudaPeekAtLastError() );
        */
    }

    // Kernel; PQ = exp(PQ)
    for (int i = 0; i < num_streams; i++) {
        int num_block_sentences = i == num_streams - 1 ?
                num_block_sentences_base + num_sentences % num_streams :
                num_block_sentences_base;
        dim3 dimBlock(block_size);
        dim3 dimGrid(MAX(1, num_block_sentences / dimBlock.x));
        g_exp<<<dimGrid, dimBlock, 0, streams[i]>>>(
                d_PQ + num_block_sentences_base * num_questions * i,
                num_block_sentences * num_questions);
    }

#if ZERO_SKIP
    // Kernel; sparse PQ; https://stackoverflow.com/a/48449394
    for (int i = 0; i < num_streams; i++) {
        int num_block_sentences = i == num_streams - 1 ?
                num_block_sentences_base + num_sentences % num_streams :
                num_block_sentences_base;

        // 1. Configure vars
        cusparseMatDescr_t desc_PQ;
        CUSPARSE_CALL( cusparseCreateMatDescr(&desc_PQ) );
        CUSPARSE_CALL( cusparseSetMatIndexBase(desc_PQ, CUSPARSE_INDEX_BASE_ZERO) );
        CUSPARSE_CALL( cusparseSetMatType(desc_PQ, CUSPARSE_MATRIX_TYPE_GENERAL) );

        int *csrr_PQ, *csrc_PQ;
        CUDA_CALL( cudaMalloc(&csrr_PQ, sizeof(int) * (num_block_sentences + 1)) );

        // 2. Query workspace
        float threshold = 0.01;
        float *csrv_PQ;
        size_t sz_work;
        CUSPARSE_CALL(cusparseSpruneDense2csr_bufferSizeExt(cusparse_handles[i],
                num_block_sentences, num_questions,
                d_PQ + num_block_sentences_base * num_questions * i, num_block_sentences,
                &threshold,
                desc_PQ,
                csrv_PQ,
                csrr_PQ,
                csrc_PQ,
                &sz_work));

        // 3. Compute csrr and nnz
        float *d_work;
        CUDA_CALL( cudaMalloc(&d_work, sz_work) );

        int nnz_PQ;
        CUSPARSE_CALL(cusparseSpruneDense2csrNnz(cusparse_handles[i],
                num_block_sentences, num_questions,
                d_PQ + num_block_sentences_base * num_questions * i, num_block_sentences,
                &threshold,
                desc_PQ,
                csrr_PQ,
                &nnz_PQ,
                d_work));

        // 4. Compute csrc, csrv
        CUDA_CALL( cudaMalloc(&csrc_PQ, sizeof(int) * nnz_PQ) );
        CUDA_CALL( cudaMalloc(&csrv_PQ, sizeof(float) * nnz_PQ) );

        CUSPARSE_CALL(cusparseSpruneDense2csr(cusparse_handles[i],
                num_block_sentences, num_questions,
                d_PQ + num_block_sentences_base * num_questions * i, num_block_sentences,
                &threshold,
                desc_PQ,
                csrv_PQ,
                csrr_PQ,
                csrc_PQ,
                d_work));

        // 5. Free
        CUDA_CALL( cudaFree(csrr_PQ) );
        CUDA_CALL( cudaFree(csrc_PQ) );
        CUDA_CALL( cudaFree(csrv_PQ) );
        CUDA_CALL( cudaFree(d_work) );
        CUSPARSE_CALL( cusparseDestroyMatDescr(desc_PQ) );

        /*
        // 1. Calc num non zero (nnz)
        thrust::device_ptr<float> t_PQ = thrust::device_pointer_cast(
                d_PQ + num_block_sentences_base * num_questions * i);
        int nz_PQ = thrust::count(thrust::cuda::par.on(streams[i]),
                t_PQ, t_PQ + num_block_sentences * num_questions, 0);
        int nnz_PQ = num_block_sentences * num_questions - nz_PQ;
        // 2. Calc nnz per row
        int *nnzr_PQ;
        CUDA_CALL( cudaMalloc(&nnzr_PQ, sizeof(int) * num_block_sentences) );
        cusparseMatDescr_t desc_PQ;
        CUSPARSE_CALL( cusparseCreateMatDescr(&desc_PQ) );
        CUSPARSE_CALL(cusparseSnnz(cusparse_handles[i],
                CUSPARSE_DIRECTION_ROW,
                num_block_sentences,
                num_questions,
                desc_PQ,
                d_PQ + num_block_sentences_base * num_questions * i, num_block_sentences,
                nnzr_PQ,
                &nnz_PQ));
        // 3. Perform dense2csr
        float *csrv_PQ;
        int *csrr_PQ, *csrc_PQ;
        CUDA_CALL( cudaMalloc(&csrv_PQ, sizeof(int) * nnz_PQ) );
        CUDA_CALL( cudaMalloc(&csrr_PQ, sizeof(int) * (num_block_sentences + 1)) );
        CUDA_CALL( cudaMalloc(&csrc_PQ, sizeof(int) * nnz_PQ) );
        CUSPARSE_CALL(cusparseSdense2csr(cusparse_handles[i],
                num_block_sentences,
                num_questions,
                desc_PQ,
                d_PQ + num_block_sentences_base * num_questions * i, num_block_sentences,
                nnzr_PQ,
                csrv_PQ,
                csrr_PQ,
                csrc_PQ));
        CUSPARSE_CALL( cusparseDestroyMatDescr(desc_PQ) );
        CUDA_CALL( cudaFree(csrc_PQ) );
        CUDA_CALL( cudaFree(csrr_PQ) );
        CUDA_CALL( cudaFree(csrv_PQ) );
        CUDA_CALL( cudaFree(nnzr_PQ) );
        */
    }
#endif

    // Kernel; compact
    /*
    for (int j = 0; j < num_questions; j++) {
        for (int i = 0; i < num_streams; i++) {
            int num_block_sentences = i == num_streams - 1 ?
                    num_block_sentences_base + num_sentences % num_streams :
                    num_block_sentences_base;

            thrust::device_ptr<float> t_PQ = thrust::device_pointer_cast(
                    d_PQ + num_block_sentences_base * num_questions * i);
            typedef thrust::device_vector<float>::iterator Iterator;
            strided_range<Iterator> t_P(
                    t_PQ,
                    t_PQ + num_block_sentences * num_questions,
                    num_questions);
            thrust::device_vector<float> t_Q;
            t_Q.reserve(num_block_sentences);
            thrust::copy_if(
                    thrust::cuda::par.on(streams[i]),
                    t_P.begin(), t_P.end(), t_Q.begin(), is_skipped());
        }
    }
    */

    // Kernel; S(i) = sum(i'th col of PQ);
    for (int i = 0; i < num_streams; i++) {
        int num_block_sentences = i == num_streams - 1 ?
                num_block_sentences_base + num_sentences % num_streams :
                num_block_sentences_base;
        float alpha = 1., beta = 0.;
        CUBLAS_CALL(cublasSgemv(cublas_handles[i],
                CUBLAS_OP_T,
                num_block_sentences, num_questions,
                &alpha,
                d_PQ + num_block_sentences_base * num_questions * i, num_block_sentences,
                d_OS, 1,
                &beta,
                d_SS + num_questions * i, 1));
    }

    // EOM to device
    for (int i = 0; i < num_streams; i++) {
        int num_block_sentences = i == num_streams - 1 ?
                num_block_sentences_base + num_sentences % num_streams :
                num_block_sentences_base;
        int size = sizeof(float) * num_block_sentences * embed_dim;;
        int offset = num_block_sentences_base * embed_dim * i;
        CUDA_CALL( cudaMemcpyAsync(d_EIM + offset, h_EIM + offset, size, cudaMemcpyHostToDevice, streams[i]) );
    }

    // Kernel; SOM = tr(PQ) * EOM
    for (int i = 0; i < num_streams; i++) {
        int num_block_sentences = i == num_streams - 1 ?
                num_block_sentences_base + num_sentences % num_streams :
                num_block_sentences_base;
        float alpha = 1., beta = 0.;
        // Do the actual multiplication
        CUBLAS_CALL(cublasSgemm(cublas_handles[i],
                CUBLAS_OP_T, CUBLAS_OP_N,
                num_questions, embed_dim, num_block_sentences,
                &alpha,
                d_PQ + num_block_sentences_base * num_questions * i, num_block_sentences,
                d_EOM + num_block_sentences_base * embed_dim * i, num_block_sentences,
                &beta,
                d_SOM + num_questions * embed_dim * i, num_questions));
    }

    CUDA_CALL( cudaDeviceSynchronize() );

    // Kernel; merge SS
    {
        float alpha = 1., beta = 0.;
        // Do the actual multiplication
        CUBLAS_CALL(cublasSgemm(cublas_handles[0],
                CUBLAS_OP_N, CUBLAS_OP_N,
                1, num_questions, num_streams,
                &alpha,
                d_OS, 1,
                d_SS, num_streams,
                &beta,
                d_S, 1));
        CUDA_CALL( cudaPeekAtLastError() );
    }

    // Kernel; merge SOM
    {
        float alpha = 1., beta = 0.;
        // Do the actual multiplication
        CUBLAS_CALL(cublasSgemm(cublas_handles[0],
                CUBLAS_OP_N, CUBLAS_OP_N,
                1, num_questions * embed_dim, num_streams,
                &alpha,
                d_OS, 1,
                d_SOM, num_streams,
                &beta,
                d_OM, 1));
        CUDA_CALL( cudaPeekAtLastError() );
    }

    // Kernel; divide OM by S
    {
        dim3 dimBlock(block_size);
        dim3 dimGrid(MAX(1, num_questions * embed_dim / dimBlock.x));
        g_normalize<<<dimGrid, dimBlock, 0, streams[0]>>>(
                d_OM, d_S,
                embed_dim, num_questions * embed_dim);
        CUDA_CALL( cudaPeekAtLastError() );
    }

    // Get the final result
    size = sizeof(float) * num_questions * embed_dim;
    CUDA_CALL( cudaMemcpyAsync(h_OM, d_OM, size, cudaMemcpyDeviceToHost, streams[0]) );

    CUDA_CALL( cudaDeviceSynchronize() );
    gettimeofday(&tv_end, NULL);
    long l_begin = (long)tv_begin.tv_sec * 1000 + (long)tv_begin.tv_usec / 1000;
    long l_end = (long)tv_end.tv_sec * 1000 + (long)tv_end.tv_usec / 1000;

    // Destroy CUDA streams, cublas contexts
    for (int i = 0; i < num_streams; i++) {
        CUSPARSE_CALL( cusparseDestroy(cusparse_handles[i]) );
        CUBLAS_CALL( cublasDestroy(cublas_handles[i]) );
        cudaStreamDestroy(streams[i]);
    }
    printf("%ld ms\n", (l_end - l_begin));

    CUDA_CALL( cudaFreeHost(h_EIM) );
    CUDA_CALL( cudaFree(d_EIM) );
    CUDA_CALL( cudaFreeHost(h_EQ) );
    CUDA_CALL( cudaFree(d_EQ) );
    CUDA_CALL( cudaFree(d_PQ) );
    CUDA_CALL( cudaFree(d_MQ) );
    CUDA_CALL( cudaFree(d_SS) );
    CUDA_CALL( cudaFreeHost(h_OS) );
    CUDA_CALL( cudaFree(d_OS) );
    CUDA_CALL( cudaFreeHost(h_EOM) );
    CUDA_CALL( cudaFree(d_SOM) );
    CUDA_CALL( cudaFreeHost(h_OM) );
    CUDA_CALL( cudaFree(d_OM) );

    CUDA_CALL( cudaDeviceReset() );

    printf("main end\n"); fflush(stdout);

    return 0;
}
