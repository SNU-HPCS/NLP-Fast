#ifndef __MATMUL_H__
#define __MATMUL_H__

#include <cuda.h>
#include <assert.h>

#define CEILING(X, Y)   (((X) + (Y) - 1) / (Y))

template <int block_size_y, int block_size_x, int loop_len>
__global__ void g_matmul(const float *A, const float *B, float *C,
        const int m, const int k, const int n)
{
    extern __shared__ float shared_mem[];
    float *s_A = shared_mem; // block_size_y * loop_len
    float *s_B = s_A + block_size_y * loop_len; // loop_len * block_size_x

    int ry = block_size_y * blockIdx.x + threadIdx.x;
    int rx = block_size_x * blockIdx.y + threadIdx.y;
    float s = 0.;
    for (int bi = 0; bi < CEILING(k, loop_len); bi++) {
        // load s_A
        if (ry < m) {
            for (int si = 0; si < CEILING(loop_len, block_size_x); si++) {
                int y = threadIdx.x;
                int x = block_size_x * si + threadIdx.y;
                int sx = loop_len * bi + x;
                if (x < loop_len && sx < k) {
                    s_A[loop_len * y + x] = A[k * ry + sx];
                }
            }
        }
        // load s_B
        if (rx < n) {
            for (int si = 0; si < CEILING(loop_len, block_size_y); si++) {
                int y = block_size_y * si + threadIdx.x;
                int x = threadIdx.y;
                int sy = loop_len * bi + y;
                if (y < loop_len && sy < k) {
                    s_B[block_size_x * y + x] = B[n * sy + rx];
                }
            }
        }
        __syncthreads();
        // c = s_A * s_B
        if (ry < m && rx < n) {
            for (int li = 0; li < loop_len; li++)
                s += s_A[loop_len * threadIdx.x + li] * s_B[block_size_x * li + threadIdx.y];
        }
        __syncthreads();
    }
    if (ry < m && rx < n)
        C[n * ry + rx] = s;
}

template <int block_size_y, int block_size_x, int loop_len>
void matmul(cudaStream_t &stream,
        int m, int k, int n,
        const float *d_A, const float *d_B, float *d_C)
{
    assert(block_size_y <= m);
    assert(block_size_x <= n);
    assert(loop_len <= k);
    dim3 dimBlock(block_size_y, block_size_x);
    dim3 dimGrid(CEILING(m, dimBlock.y), CEILING(n, dimBlock.x));
    int shm_size = sizeof(float) * (dimBlock.y + dimBlock.x) * loop_len;
    g_matmul<block_size_y, block_size_x, loop_len>
            <<<dimGrid, dimBlock, shm_size, stream>>>(d_A, d_B, d_C, m, k, n);
}

template <int block_size_y, int block_size_x, int loop_len>
__global__ void g_matmul_zs(const float *A, const float *B, float *C,
        const int m, const int k, const int n)
{
    extern __shared__ float shared_mem[];
    float *s_A = shared_mem; // block_size_y * loop_len
    float *s_B = s_A + block_size_y * loop_len; // loop_len * block_size_x

    int ry = block_size_y * blockIdx.x + threadIdx.x;
    int rx = block_size_x * blockIdx.y + threadIdx.y;
    float s = 0.;
    for (int bi = 0; bi < CEILING(k, loop_len); bi++) {
        // load s_A
        if (ry < m) {
            for (int si = 0; si < CEILING(loop_len, block_size_x); si++) {
                int y = threadIdx.x;
                int x = block_size_x * si + threadIdx.y;
                int sx = loop_len * bi + x;
                if (x < loop_len && sx < k) {
                    s_A[loop_len * y + x] = A[k * ry + sx];
                }
            }
        }
        __syncthreads(); // ZS: need to ensure we are reading the init'ed s_A
        // load s_B
        if (rx < n) {
            for (int si = 0; si < CEILING(loop_len, block_size_y); si++) {
                int y = block_size_y * si + threadIdx.x;
                int x = threadIdx.y;
                int sy = loop_len * bi + y;
                if (y < loop_len && sy < k) {
                    if (s_A[loop_len * x + y] < 0.1) // ZS: condition check
                        s_A[loop_len * x + y] = 0.;  // ZS: skip loading B
                    else                             // ZS: otherwise, just as before
                        s_B[block_size_x * y + x] = B[n * sy + rx];
                }
            }
        }
        __syncthreads();
        // c = s_A * s_B
        if (ry < m && rx < n) {
            for (int li = 0; li < loop_len; li++)
                s += s_A[loop_len * threadIdx.x + li] * s_B[block_size_x * li + threadIdx.y];
        }
        __syncthreads();
    }
    if (ry < m && rx < n)
        C[n * ry + rx] = s;
}

template <int block_size_y, int block_size_x, int loop_len>
void matmul_zs(cudaStream_t &stream,
        int m, int k, int n,
        const float *d_A, const float *d_B, float *d_C)
{
    assert(block_size_y <= m);
    assert(block_size_x <= n);
    assert(loop_len <= k);
    dim3 dimBlock(block_size_y, block_size_x);
    dim3 dimGrid(CEILING(m, dimBlock.y), CEILING(n, dimBlock.x));
    int shm_size = sizeof(float) * (dimBlock.y + dimBlock.x) * loop_len;
    g_matmul_zs<block_size_y, block_size_x, loop_len>
            <<<dimGrid, dimBlock, shm_size, stream>>>(d_A, d_B, d_C, m, k, n);
}
#endif
