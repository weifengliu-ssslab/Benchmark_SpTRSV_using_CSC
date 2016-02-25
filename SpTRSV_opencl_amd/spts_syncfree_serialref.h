#ifndef _SPTS_SYNCFREE_SERIALREF_
#define _SPTS_SYNCFREE_SERIALREF_

#include "common.h"
#include "utils.h"
#include "tranpose.h"

int spts_syncfree_analyser(const int   *cscRowIdx,
                           const int    m,
                           const int    n,
                           const int    nnz,
                                 int   *csrRowHisto)
{
    memset(csrRowHisto, 0, m * sizeof(int));

    // generate row pointer by partial transposition
//#pragma omp parallel for
    for (int i = 0; i < nnz; i++)
    {
//#pragma omp atomic
        csrRowHisto[cscRowIdx[i]]++;
    }

    return 0;
}

int spts_syncfree_executor(const int    *cscColPtr,
                           const int    *cscRowIdx,
                           const VALUE_TYPE    *cscVal,
                           const int    *csrRowHisto,
                           const int    m,
                           const int    n,
                           const int     nnz,
                           const VALUE_TYPE    *b,
                                 VALUE_TYPE    *x)
{
    // malloc tmp memory to simulate atomic operations
    int *csrRowHisto_atomic = (int *)malloc(sizeof(int) * m);
    memset(csrRowHisto_atomic, 0, sizeof(int) * m);

    // malloc tmp memory to collect a partial sum of each row
    VALUE_TYPE *left_sum = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m);
    memset(left_sum, 0, sizeof(VALUE_TYPE) * m);

    for (int i = 0; i < n; i++)
    {
        int dia = csrRowHisto[i] - 1;

        // while loop, i.e., wait, until all nnzs are prepared
        do
        {
            // just wait
        }
        while (dia != csrRowHisto_atomic[i]);

        VALUE_TYPE xi = (b[i] - left_sum[i]) / cscVal[cscColPtr[i]];
        x[i] = xi;

        for (int j = cscColPtr[i] + 1; j < cscColPtr[i+1]; j++)
        {
            int rowIdx = cscRowIdx[j];
            // atomic add
            left_sum[rowIdx] += xi * cscVal[j];
            csrRowHisto_atomic[rowIdx] += 1;
        }
    }

    free(csrRowHisto_atomic);
    free(left_sum);

    return 0;
}

int spts_syncfree_serialref(const int           *csrRowPtrL_tmp,
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

    int *cscRowIdxL = (int *)malloc(nnzL * sizeof(int));
    int *cscColPtrL = (int *)malloc((n+1) * sizeof(int));
    memset(cscColPtrL, 0, (n+1) * sizeof(int));
    VALUE_TYPE *cscValL    = (VALUE_TYPE *)malloc(nnzL * sizeof(VALUE_TYPE));

    printf("\n");

    struct timeval t1, t2;

    // transpose from csr to csc first
    gettimeofday(&t1, NULL);

    for (int i = 0; i < BENCH_REPEAT; i++)
        matrix_transposition(m, n, nnzL,
                         csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp,
                         cscRowIdxL, cscColPtrL, cscValL);

    gettimeofday(&t2, NULL);
    double time_trans = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_trans /= BENCH_REPEAT;

    //  - SpTS Serial analyser start!
    printf(" - SpTS Serial analyser start!\n");

    gettimeofday(&t1, NULL);

    int *csrRowHistoL = (int *)malloc(m * sizeof(int));

    for (int i = 0; i < BENCH_REPEAT; i++)
        spts_syncfree_analyser(cscRowIdxL, m, n, nnzL, csrRowHistoL);

    gettimeofday(&t2, NULL);
    double time_spts_analyser = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_spts_analyser /= BENCH_REPEAT;
    printf("SpTS Serial analyser on L used %4.2f ms\n", time_spts_analyser);

    // validate csrRowHistoL
    int err_counter = 0;
    for (int i = 0; i < m; i++)
    {
        if (csrRowHistoL[i] != csrRowPtrL_tmp[i+1] - csrRowPtrL_tmp[i])
            err_counter++;
    }

    if (!err_counter)
        printf("SpTS Serial analyser on L passed!\n");
    else
        printf("SpTS Serial analyser on L failed!\n");

    VALUE_TYPE *x_ref = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n);
    for ( int i = 0; i < n; i++)
        x_ref[i] = 1;

    VALUE_TYPE *b = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m);

    // run spmv to get b
    gettimeofday(&t1, NULL);

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        for (int i = 0; i < m; i++)
        {
            b[i] = 0;
            for (int j = csrRowPtrL_tmp[i]; j < csrRowPtrL_tmp[i+1]; j++)
                b[i] += csrValL_tmp[j] * x_ref[csrColIdxL_tmp[j]];
        }
    }

    gettimeofday(&t2, NULL);
    double time_spmv = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_spmv /= BENCH_REPEAT;
    
    VALUE_TYPE *x = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n);

    //  - SpTS Serial executor start!
    printf(" - SpTS Serial executor start!\n");

    gettimeofday(&t1, NULL);

    for (int i = 0; i < BENCH_REPEAT; i++)
        spts_syncfree_executor(cscColPtrL, cscRowIdxL, cscValL, csrRowHistoL, m, n, nnzL, b, x);

    gettimeofday(&t2, NULL);
    double time_spts_executor = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_spts_executor /= BENCH_REPEAT;
    printf("SpTS Serial executor used %4.2f ms, throughput is %4.2f gflops\n",
           time_spts_executor, 2*nnzL/(1e6*time_spts_executor));

    // validate x
    err_counter = 0;
    for (int i = 0; i < m; i++)
    {
        if (x[i] != x_ref[i])
            err_counter++;
    }

    if (!err_counter)
        printf("SpTS Serial executor on L passed!\n");
    else
        printf("SpTS Serial executor on L failed!\n");

    // print SpMV throughput
    printf("\nAs a reference, L's SpMV used %4.2f ms, throughput is %4.2f gflops\n",
           time_spmv, 2*nnzL/(1e6*time_spmv));
    printf("As a reference, L's transposition (csr->csc) used %4.2f ms\n", time_trans);

    free(csrRowHistoL);

    free(cscRowIdxL);
    free(cscColPtrL);
    free(cscValL);

    free(x);
    free(x_ref);
    free(b);

    return 0;
}


#endif


