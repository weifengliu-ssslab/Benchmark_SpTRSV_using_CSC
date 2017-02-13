#ifndef _SPTS_SYNCFREE_CUDA_
#define _SPTS_SYNCFREE_CUDA_

#include "common.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

__global__
void spts_syncfree_cuda_analyser(const int   *d_cscRowIdx,
                                 const int    m,
                                 const int    nnz,
                                       int   *d_csrRowHisto)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);
    if (global_id < nnz)
    {
        atomicAdd(&d_csrRowHisto[d_cscRowIdx[global_id]], 1);
    }
}

__global__
void spts_syncfree_cuda_executor_pre(const int   *d_csrRowPtrL,
                                     const int    m,
                                           int   *d_csrRowHisto)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);
    if (global_id < m)
    {
        d_csrRowHisto[global_id] = d_csrRowPtrL[global_id+1] - d_csrRowPtrL[global_id];
    }
}

__global__
void spts_syncfree_cuda_executor(const int* __restrict__        d_cscColPtr,
                                 const int* __restrict__        d_cscRowIdx,
                                 const VALUE_TYPE* __restrict__ d_cscVal,
                                 const int* __restrict__        d_csrRowPtr,
                                 int*                           d_csrRowHisto,
                                 VALUE_TYPE*                    d_left_sum,
                                 VALUE_TYPE*                    d_partial_sum,
                                 const int                      m,
                                 const int                      nnz,
                                 const VALUE_TYPE* __restrict__ d_b,
                                 VALUE_TYPE*                    d_x,
                                 int*                           d_while_profiler)

{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_x_id = global_id / WARP_SIZE;
    volatile __shared__ int s_csrRowHisto[WARP_PER_BLOCK];
    volatile __shared__ VALUE_TYPE s_left_sum[WARP_PER_BLOCK];

    if (global_x_id >= m) return;
    // Initialize
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int starting_x = (global_id / (WARP_PER_BLOCK * WARP_SIZE)) * WARP_PER_BLOCK;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    // Prefetch
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[d_cscColPtr[global_x_id]];
    //asm("prefetch.global.L2 [%0];"::"d"(d_cscVal[d_cscColPtr[global_x_id] + 1 + lane_id]));
    asm("prefetch.global.L2 [%0];"::"r"(d_cscRowIdx[d_cscColPtr[global_x_id] + 1 + lane_id]));

    if (threadIdx.x < WARP_PER_BLOCK) { s_csrRowHisto[threadIdx.x] = 1; s_left_sum[threadIdx.x] = 0; }
    __syncthreads();

    //clock_t start;
    // Consumer
    //do {
    //    start = clock();
    //}
    //while (s_csrRowHisto[local_warp_id] != d_csrRowHisto[global_x_id]);
  
    // Consumer (fixed a problem that happens on Tesla P100)
    int graphInDegree;
    do {
        //bypass Tex cache and avoid other mem optimization by nvcc/ptxas
        asm("ld.global.u32 %0, [%1];" : "=r"(graphInDegree),"=r"(d_csrRowHisto[global_x_id]) :: "memory"); 
    }
    while (s_csrRowHisto[local_warp_id] != graphInDegree );

    VALUE_TYPE xi = d_left_sum[global_x_id] + s_left_sum[local_warp_id]; 
    xi = (d_b[global_x_id] - xi) * coef;

    // Producer
    for (int j = d_cscColPtr[global_x_id] + 1 + lane_id; j < d_cscColPtr[global_x_id+1]; j += WARP_SIZE) {   
        int rowIdx = d_cscRowIdx[j];
        if (rowIdx < starting_x + WARP_PER_BLOCK) {
            atomicAdd((VALUE_TYPE *)&s_left_sum[rowIdx - starting_x], xi * d_cscVal[j]);
            atomicAdd((int *)&s_csrRowHisto[rowIdx - starting_x], 1);
        }
        else {
            atomicAdd(&d_left_sum[rowIdx], xi * d_cscVal[j]);
            atomicSub(&d_csrRowHisto[rowIdx], 1);
        }
    }
    // Finish
    if (!lane_id) d_x[global_x_id] = xi;
}

int spts_syncfree_cuda(const int           *csrRowPtrL_tmp,
                          const int           *csrColIdxL_tmp,
                          const VALUE_TYPE    *csrValL_tmp,
                          const int            m,
                          const int            n,
                          const int            nnzL)
{
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }

    VALUE_TYPE *x_ref = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n);
    for ( int i = 0; i < n; i++)
        x_ref[i] = 1;

    VALUE_TYPE *b = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m);

    for (int i = 0; i < m; i++)
    {
        b[i] = 0;
        for (int j = csrRowPtrL_tmp[i]; j < csrRowPtrL_tmp[i+1]; j++)
            b[i] += csrValL_tmp[j] * x_ref[csrColIdxL_tmp[j]];
    }

    VALUE_TYPE *x = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n);

    // transpose from csr to csc first
    int *cscRowIdxL = (int *)malloc(nnzL * sizeof(int));
    int *cscColPtrL = (int *)malloc((n+1) * sizeof(int));
    memset(cscColPtrL, 0, (n+1) * sizeof(int));
    VALUE_TYPE *cscValL    = (VALUE_TYPE *)malloc(nnzL * sizeof(VALUE_TYPE));

    matrix_transposition(m, n, nnzL,
                         csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp,
                         cscRowIdxL, cscColPtrL, cscValL);

    // transfer host mem to device mem
    int *d_cscColPtrL;
    int *d_cscRowIdxL;
    VALUE_TYPE *d_cscValL;
    VALUE_TYPE *d_b;
    VALUE_TYPE *d_x;

    // Matrix L
    cudaMalloc((void **)&d_cscColPtrL, (n+1) * sizeof(int));
    cudaMalloc((void **)&d_cscRowIdxL, nnzL  * sizeof(int));
    cudaMalloc((void **)&d_cscValL,    nnzL  * sizeof(VALUE_TYPE));

    cudaMemcpy(d_cscColPtrL, cscColPtrL, (n+1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscRowIdxL, cscRowIdxL, nnzL  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscValL,    cscValL,    nnzL  * sizeof(VALUE_TYPE),   cudaMemcpyHostToDevice);

    // Vector b
    cudaMalloc((void **)&d_b, m * sizeof(VALUE_TYPE));
    cudaMemcpy(d_b, b, m * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

    // Vector x
    cudaMalloc((void **)&d_x, n  * sizeof(VALUE_TYPE));
    cudaMemset(d_x, 0, n * sizeof(VALUE_TYPE));

    //  - cuda syncfree SpTS analysis start!
    printf(" - cuda syncfree SpTS analysis start!\n");

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    // malloc tmp memory to simulate atomic operations
    int *d_csrRowHisto;
    cudaMalloc((void **)&d_csrRowHisto, sizeof(int) * (m+1));

    // generate row pointer by partial transposition
    int *d_csrRowPtrL;
    cudaMalloc((void **)&d_csrRowPtrL, (m+1) * sizeof(int));
    thrust::device_ptr<int> d_csrRowPtrL_thrust = thrust::device_pointer_cast(d_csrRowPtrL);
    thrust::device_ptr<int> d_csrRowHisto_thrust = thrust::device_pointer_cast(d_csrRowHisto);

    // malloc tmp memory to collect a partial sum of each row
    VALUE_TYPE *d_left_sum;
    cudaMalloc((void **)&d_left_sum, sizeof(VALUE_TYPE) * m);

    // malloc tmp memory to collect a partial sum of each row
    VALUE_TYPE *d_partial_sum;
    cudaMalloc((void **)&d_partial_sum, sizeof(VALUE_TYPE) * nnzL);
    //cudaMemset(d_partial_sum, 0, sizeof(VALUE_TYPE) * nnzL);

    int num_threads = 256;
    int num_blocks = ceil ((double)nnzL / (double)num_threads);

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        cudaMemset(d_csrRowHisto, 0, (m+1) * sizeof(int));
        spts_syncfree_cuda_analyser<<< num_blocks, num_threads >>>
                                      (d_cscRowIdxL, m, nnzL, d_csrRowHisto);
    }
    cudaDeviceSynchronize();

    gettimeofday(&t2, NULL);
    double time_cuda_analysis = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_analysis /= BENCH_REPEAT;

    thrust::exclusive_scan(d_csrRowHisto_thrust, d_csrRowHisto_thrust + m+1, d_csrRowPtrL_thrust);

    printf("cuda syncfree SpTS analysis on L used %4.2f ms\n", time_cuda_analysis);

    // validate csrRowPtrL
    int *csrRowPtrL = (int *)malloc((m+1) * sizeof(int));
    cudaMemcpy(csrRowPtrL, d_csrRowPtrL, (m+1) * sizeof(int), cudaMemcpyDeviceToHost);

    int err_counter = 0;
    for (int i = 0; i <= m; i++)
    {
        //printf("[%i]: csrRowPtrL = %i, csrRowPtrL_tmp = %i\n", i, csrRowPtrL[i], csrRowPtrL_tmp[i]);
        if (csrRowPtrL[i] != csrRowPtrL_tmp[i])
            err_counter++;
    }

    free(csrRowPtrL);

    if (!err_counter)
        printf("cuda syncfree SpTS analyser on L passed!\n");
    else
        printf("cuda syncfree SpTS analyser on L failed!\n");

    //  - cuda syncfree SpTS solve start!
    printf(" - cuda syncfree SpTS solve start!\n");

    int *d_while_profiler;
    cudaMalloc((void **)&d_while_profiler, sizeof(int) * n);
    cudaMemset(d_while_profiler, 0, sizeof(int) * n);
    int *while_profiler = (int *)malloc(sizeof(int) * n);

    // step 5: solve L*y = x
    double time_cuda_solve = 0;

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        num_threads = 256;
        num_blocks = ceil ((double)m / (double)(num_threads));
        spts_syncfree_cuda_executor_pre<<< num_blocks, num_threads >>>
                                          (d_csrRowPtrL, m, d_csrRowHisto);
        
        gettimeofday(&t1, NULL);

        cudaMemset(d_left_sum, 0, sizeof(VALUE_TYPE) * m);

        num_threads = WARP_PER_BLOCK * WARP_SIZE;
        num_blocks = ceil ((double)m / (double)(num_threads/WARP_SIZE));

        spts_syncfree_cuda_executor<<< num_blocks, num_threads >>>
                                   (d_cscColPtrL, d_cscRowIdxL, d_cscValL, 
                                    d_csrRowPtrL, d_csrRowHisto, 
                                    d_left_sum, d_partial_sum,
                                    m, nnzL, d_b, d_x, d_while_profiler);
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);

        time_cuda_solve += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    }
    cudaDeviceSynchronize();

    time_cuda_solve /= BENCH_REPEAT;

    printf("cuda syncfree SpTS solve used %4.2f ms, throughput is %4.2f gflops\n",
           time_cuda_solve, 2*nnzL/(1e6*time_cuda_solve));

    cudaMemcpy(x, d_x, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    // validate x
    err_counter = 0;
    for (int i = 0; i < n; i++)
    {
        if (abs(x_ref[i] - x[i]) > 0.01 * abs(x_ref[i]))
            err_counter++;
    }

    if (!err_counter)
        printf("cuda syncfree SpTS on L passed!\n");
    else
        printf("cuda syncfree SpTS on L failed!\n");

    cudaMemcpy(while_profiler, d_while_profiler, n * sizeof(int), cudaMemcpyDeviceToHost);
    long long unsigned int while_count = 0;
    for (int i = 0; i < n; i++)
    {
        while_count += while_profiler[i];
        //printf("while_profiler[%i] = %i\n", i, while_profiler[i]);
    }
    //printf("\nwhile_count= %llu in total, %llu per row/column\n", while_count, while_count/m);

    // step 6: free resources
    free(while_profiler);

    cudaFree(d_csrRowHisto);
    cudaFree(d_left_sum);
    cudaFree(d_partial_sum);
    cudaFree(d_csrRowPtrL);
    cudaFree(d_while_profiler);

    cudaFree(d_cscColPtrL);
    cudaFree(d_cscRowIdxL);
    cudaFree(d_cscValL);
    cudaFree(d_b);
    cudaFree(d_x);

    return 0;
}

#endif



