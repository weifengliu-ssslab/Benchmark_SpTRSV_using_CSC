#ifndef _SPTS_SYNCFREE_OPENCL_
#define _SPTS_SYNCFREE_OEPNCL_

#include "common.h"
#include "utils.h"
#include "basiccl.h"

int spts_syncfree_opencl (const int           *csrRowPtrL_tmp,
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

    int err = 0;

    // set device
    BasicCL basicCL;
    cl_event            ceTimer;                 // OpenCL event
    cl_ulong            queuedTime;
    cl_ulong            submitTime;
    cl_ulong            startTime;
    cl_ulong            endTime;

    char platformVendor[CL_STRING_LENGTH];
    char platformVersion[CL_STRING_LENGTH];

    char gpuDeviceName[CL_STRING_LENGTH];
    char gpuDeviceVersion[CL_STRING_LENGTH];
    int  gpuDeviceComputeUnits;
    cl_ulong  gpuDeviceGlobalMem;
    cl_ulong  gpuDeviceLocalMem;

    cl_uint             numPlatforms;           // OpenCL platform
    cl_platform_id*     cpPlatforms;

    cl_uint             numGpuDevices;          // OpenCL Gpu device
    cl_device_id*       cdGpuDevices;

    cl_context          cxGpuContext;           // OpenCL Gpu context
    cl_command_queue    ocl_command_queue;      // OpenCL Gpu command queues

    bool profiling = true;
    int select_device = 2;

    // platform
    err = basicCL.getNumPlatform(&numPlatforms);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    printf("platform number: %i.\n", numPlatforms);

    cpPlatforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);

    err = basicCL.getPlatformIDs(cpPlatforms, numPlatforms);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    for (unsigned int i = 0; i < numPlatforms; i++)
    {
        err = basicCL.getPlatformInfo(cpPlatforms[i], platformVendor, platformVersion);
        if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

        // Gpu device
        err = basicCL.getNumGpuDevices(cpPlatforms[i], &numGpuDevices);

        if (numGpuDevices > 0)
        {
            cdGpuDevices = (cl_device_id *)malloc(numGpuDevices * sizeof(cl_device_id) );

            err |= basicCL.getGpuDeviceIDs(cpPlatforms[i], numGpuDevices, cdGpuDevices);

            err |= basicCL.getDeviceInfo(cdGpuDevices[select_device], gpuDeviceName, gpuDeviceVersion,
                                         &gpuDeviceComputeUnits, &gpuDeviceGlobalMem,
                                         &gpuDeviceLocalMem, NULL);
            if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

            printf("Platform [%i] Vendor: %s Version: %s\n", i, platformVendor, platformVersion);
            printf("Using GPU device: %s ( %i CUs, %lu kB local, %lu MB global, %s )\n",
                   gpuDeviceName, gpuDeviceComputeUnits,
                   gpuDeviceLocalMem / 1024, gpuDeviceGlobalMem / (1024 * 1024), gpuDeviceVersion);

            break;
        }
        else
        {
            continue;
        }
    }

    // Gpu context
    err = basicCL.getContext(&cxGpuContext, cdGpuDevices, numGpuDevices);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // Gpu commandqueue
    if (profiling)
        err = basicCL.getCommandQueueProfilingEnable(&ocl_command_queue, cxGpuContext, cdGpuDevices[select_device]);
    else
        err = basicCL.getCommandQueue(&ocl_command_queue, cxGpuContext, cdGpuDevices[select_device]);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    const char *ocl_source_code_spts =
    "    #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable                                          \n"
    "    #pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable                                      \n"
    "    #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable                                                 \n"
    "    #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable                                             \n"
    "                                                                                                                \n"
    "    #pragma OPENCL EXTENSION cl_khr_fp64 : enable                                                               \n"
    "                                                                                                                \n"
    "    #ifndef VALUE_TYPE                                                                                          \n"
    "    #define VALUE_TYPE float                                                                                    \n"
    "    #endif                                                                                                      \n"
    "    #define WARP_SIZE 64                                                                                        \n"
    "                                                                                                                \n"
    "    inline                                                                                                      \n"
    "    void atom_add_d_fp32(volatile __global float *val,                                                          \n"
    "                       float delta)                                                                             \n"
    "    {                                                                                                           \n"
    "        union { float f; unsigned int i; } old;                                                                 \n"
    "        union { float f; unsigned int i; } new;                                                                 \n"
    "        do                                                                                                      \n"
    "        {                                                                                                       \n"
    "            old.f = *val;                                                                                       \n"
    "            new.f = old.f + delta;                                                                              \n"
    "        }                                                                                                       \n"
    "        while (atomic_cmpxchg((volatile __global unsigned int *)val, old.i, new.i) != old.i);                   \n"
    "    }                                                                                                           \n"
    "                                                                                                                \n"
    "    inline                                                                                                      \n"
    "    void atom_add_d_fp64(volatile __global double *val,                                                         \n"
    "                       double delta)                                                                            \n"
    "    {                                                                                                           \n"
    "        union { double f; ulong i; } old;                                                                       \n"
    "        union { double f; ulong i; } new;                                                                       \n"
    "        do                                                                                                      \n"
    "        {                                                                                                       \n"
    "            old.f = *val;                                                                                       \n"
    "            new.f = old.f + delta;                                                                              \n"
    "        }                                                                                                       \n"
    "        while (atom_cmpxchg((volatile __global ulong *)val, old.i, new.i) != old.i);                            \n"
    "    }                                                                                                           \n"
    "    inline                                                                                                      \n"
    "    void atom_add_s_fp32(volatile __local float *val,                                                           \n"
    "                       float delta)                                                                             \n"
    "    {                                                                                                           \n"
    "        union { float f; unsigned int i; } old;                                                                 \n"
    "        union { float f; unsigned int i; } new;                                                                 \n"
    "        do                                                                                                      \n"
    "        {                                                                                                       \n"
    "            old.f = *val;                                                                                       \n"
    "            new.f = old.f + delta;                                                                              \n"
    "        }                                                                                                       \n"
    "        while (atomic_cmpxchg((volatile __local unsigned int *)val, old.i, new.i) != old.i);                    \n"
    "    }                                                                                                           \n"
    "                                                                                                                \n"
    "    inline                                                                                                      \n"
    "    void atom_add_s_fp64(volatile __local double *val,                                                          \n"
    "                       double delta)                                                                            \n"
    "    {                                                                                                           \n"
    "        union { double f; ulong i; } old;                                                                       \n"
    "        union { double f; ulong i; } new;                                                                       \n"
    "        do                                                                                                      \n"
    "        {                                                                                                       \n"
    "            old.f = *val;                                                                                       \n"
    "            new.f = old.f + delta;                                                                              \n"
    "        }                                                                                                       \n"
    "        while (atom_cmpxchg((volatile __local ulong *)val, old.i, new.i) != old.i);                             \n"
    "    }                                                                                                           \n"
    "                                                                                                                \n"
    "    __kernel                                                                                                    \n"
    "    void spts_syncfree_opencl_analyser(__global const int      *d_cscRowIdx,                                    \n"
    "                                          const int                m,                                           \n"
    "                                          const int                nnz,                                         \n"
    "                                          __global int            *d_csrRowHisto)                               \n"
    "    {                                                                                                           \n"
    "        const int global_id = get_global_id(0);                                                                 \n"
    "        if (global_id < nnz)                                                                                    \n"
    "        {                                                                                                       \n"
    "            atomic_fetch_add_explicit((atomic_int*)&d_csrRowHisto[d_cscRowIdx[global_id]], 1,                   \n"
    "                                      memory_order_acq_rel, memory_scope_device);                               \n"
    "        }                                                                                                       \n"
    "    }                                                                                                           \n"
    "                                                                                                                \n"
    "    __kernel                                                                                                    \n"
    "    void spts_syncfree_opencl_executor(__global const int          *d_cscColPtr,                                \n"
    "                                          __global const int          *d_cscRowIdx,                             \n"
    "                                          __global const VALUE_TYPE   *d_cscVal,                                \n"
    "                                          __global volatile int       *d_csrRowHisto,                           \n"
    "                                          __global VALUE_TYPE         *d_left_sum,                              \n"
    "                                          const int                    m,                                       \n"
    "                                          const int                    nnz,                                     \n"
    "                                          __global const VALUE_TYPE   *d_b,                                     \n"
    "                                          __global VALUE_TYPE         *d_x,                                     \n"
    "                                          __local volatile int        *s_csrRowHisto,                           \n"
    "                                          __local volatile VALUE_TYPE *s_left_sum,                              \n"
    "                                          const int                    warp_per_block)                          \n"
    "    {                                                                                                           \n"
    "        const int global_id = get_global_id(0);                                                                 \n"
    "        const int local_id = get_local_id(0);                                                                   \n"
    "        const int global_x_id = global_id / WARP_SIZE;                                                          \n"
    "        if (global_x_id >= m) return;                                                                           \n"
    "                                                                                                                \n"
    "        // Initialize                                                                                           \n"
    "        const int local_warp_id = local_id / WARP_SIZE;                                                         \n"
    "        const int starting_x = (global_id / (warp_per_block * WARP_SIZE)) * warp_per_block;                     \n"
    "        const int lane_id = (WARP_SIZE - 1) & get_local_id(0);                                                  \n"
    "        if (local_id < warp_per_block) { s_csrRowHisto[local_id] = 1; s_left_sum[local_id] = 0; }               \n"
    "        barrier(CLK_LOCAL_MEM_FENCE);                                                                           \n"
    "                                                                                                                \n"
    "        // Prefetch                                                                                             \n"
    "        const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[d_cscColPtr[global_x_id]];                             \n"
    "                                                                                                                \n"
    "        // Consumer                                                                                             \n"
    "        int loads, loadd;                                                                                       \n"
    "        do {                                                                                                    \n"
    "            // busy-wait                                                                                        \n"
    "        }                                                                                                       \n"
    "        while ((loads = atomic_load_explicit((atomic_int*)&s_csrRowHisto[local_warp_id],                        \n"
    "                                             memory_order_acquire, memory_scope_work_group)) !=                 \n"
    "               (loadd = atomic_load_explicit((atomic_int*)&d_csrRowHisto[global_x_id],                          \n"
    "                                             memory_order_acquire, memory_scope_device)) );                     \n"
    "                                                                                                                \n"
    "        VALUE_TYPE xi = d_left_sum[global_x_id] + s_left_sum[local_warp_id];                                    \n"
    "        xi = (d_b[global_x_id] - xi) * coef;                                                                    \n"
    "                                                                                                                \n"
    "        // Producer                                                                                             \n"
    "        for (int j = d_cscColPtr[global_x_id] + 1 + lane_id; j < d_cscColPtr[global_x_id+1]; j += WARP_SIZE) {  \n"
    "            int rowIdx = d_cscRowIdx[j];                                                                        \n"
    "            if (rowIdx < starting_x + warp_per_block) {                                                         \n"
    "                if (sizeof(VALUE_TYPE) == 8)                                                                    \n"
    "                    atom_add_s_fp64(&s_left_sum[rowIdx - starting_x], xi * d_cscVal[j]);                        \n"
    "                else                                                                                            \n"
    "                    atom_add_s_fp32(&s_left_sum[rowIdx - starting_x], xi * d_cscVal[j]);                        \n"
    "                atomic_fetch_add_explicit((atomic_int*)&s_csrRowHisto[rowIdx - starting_x], 1,                  \n"
    "                                          memory_order_acq_rel, memory_scope_work_group);                       \n"
    "            }                                                                                                   \n"
    "            else {                                                                                              \n"
    "                if (sizeof(VALUE_TYPE) == 8)                                                                    \n"
    "                    atom_add_d_fp64(&d_left_sum[rowIdx], xi * d_cscVal[j]);                                     \n"
    "                else                                                                                            \n"
    "                    atom_add_d_fp32(&d_left_sum[rowIdx], xi * d_cscVal[j]);                                     \n"
    "                atomic_fetch_sub_explicit((atomic_int*)&d_csrRowHisto[rowIdx], 1,                               \n"
    "                                           memory_order_acq_rel, memory_scope_device);                          \n"
    "            }                                                                                                   \n"
    "        }                                                                                                       \n"
    "                                                                                                                \n"
    "        // Finish                                                                                               \n"
    "        if (!lane_id) d_x[global_x_id] = xi ;                                                                   \n"
    "    }                                                                                                           \n";

    // Create the program
    cl_program          ocl_program_spts;

    size_t source_size_spts[] = { strlen(ocl_source_code_spts)};

    ocl_program_spts = clCreateProgramWithSource(cxGpuContext, 1, &ocl_source_code_spts, source_size_spts, &err);

    if(err != CL_SUCCESS) {printf("OpenCL clCreateProgramWithSource ERROR CODE = %i\n", err); return err;}

    // Build the program

    if (sizeof(VALUE_TYPE) == 8)
        err = clBuildProgram(ocl_program_spts, 0, NULL, "-cl-std=CL2.0 -D VALUE_TYPE=double", NULL, NULL);
    else
        err = clBuildProgram(ocl_program_spts, 0, NULL, "-cl-std=CL2.0 -D VALUE_TYPE=float", NULL, NULL);
    
    // Create kernels
    cl_kernel  ocl_kernel_spts_analyser;
    cl_kernel  ocl_kernel_spts_executor;
    ocl_kernel_spts_analyser = clCreateKernel(ocl_program_spts, "spts_syncfree_opencl_analyser", &err);
    if(err != CL_SUCCESS) {printf("OpenCL clCreateKernel ERROR CODE = %i\n", err); return err;}
    ocl_kernel_spts_executor = clCreateKernel(ocl_program_spts, "spts_syncfree_opencl_executor", &err);
    if(err != CL_SUCCESS) {printf("OpenCL clCreateKernel ERROR CODE = %i\n", err); return err;}

    // transfer host mem to device mem
    // Define pointers of matrix L, vector x and b
    cl_mem      d_cscColPtrL;
    cl_mem      d_cscRowIdxL;
    cl_mem      d_cscValL;
    cl_mem      d_b;
    cl_mem      d_x;

    // Matrix L
    d_cscColPtrL = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, (n+1) * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    d_cscRowIdxL = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, nnzL  * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    d_cscValL    = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, nnzL  * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    err = clEnqueueWriteBuffer(ocl_command_queue, d_cscColPtrL, CL_TRUE, 0, (n+1) * sizeof(int), cscColPtrL, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    err = clEnqueueWriteBuffer(ocl_command_queue, d_cscRowIdxL, CL_TRUE, 0, nnzL  * sizeof(int), cscRowIdxL, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    err = clEnqueueWriteBuffer(ocl_command_queue, d_cscValL, CL_TRUE, 0, nnzL  * sizeof(VALUE_TYPE), cscValL, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // Vector b
    d_b    = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, m  * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    err = clEnqueueWriteBuffer(ocl_command_queue, d_b, CL_TRUE, 0, m  * sizeof(VALUE_TYPE), b, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // Vector x
    d_x    = clCreateBuffer(cxGpuContext, CL_MEM_READ_WRITE, n  * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    memset(x, 0, m  * sizeof(VALUE_TYPE));
    err = clEnqueueWriteBuffer(ocl_command_queue, d_x, CL_TRUE, 0, n  * sizeof(VALUE_TYPE), x, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}


    //  - opencl syncfree SpTS analysis start!
    printf(" - opencl syncfree SpTS analysis start!\n");

    // malloc tmp memory to simulate atomic operations
    cl_mem d_csrRowHisto;
    d_csrRowHisto = clCreateBuffer(cxGpuContext, CL_MEM_READ_WRITE, m * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // memset d_csrRowHisto to 0
    int *csrRowHisto = (int *)malloc(m * sizeof(int));
    memset(csrRowHisto, 0, m * sizeof(int));
    err = clEnqueueWriteBuffer(ocl_command_queue, d_csrRowHisto, CL_TRUE, 0, m  * sizeof(int), csrRowHisto, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // malloc tmp memory to collect a partial sum of each row
    cl_mem d_left_sum;
    d_left_sum = clCreateBuffer(cxGpuContext, CL_MEM_READ_WRITE, m * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // memset d_left_sum to 0
    int *left_sum = (int *)malloc(m * sizeof(VALUE_TYPE));
    memset(left_sum, 0, m * sizeof(VALUE_TYPE));
    err = clEnqueueWriteBuffer(ocl_command_queue, d_left_sum, CL_TRUE, 0, m  * sizeof(VALUE_TYPE), left_sum, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = 256;
    int num_blocks = ceil ((double)nnzL / (double)num_threads);

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    err  = clSetKernelArg(ocl_kernel_spts_analyser, 0, sizeof(cl_mem), (void*)&d_cscRowIdxL);
    err |= clSetKernelArg(ocl_kernel_spts_analyser, 1, sizeof(cl_int), (void*)&m);
    err |= clSetKernelArg(ocl_kernel_spts_analyser, 2, sizeof(cl_int), (void*)&nnzL);
    err |= clSetKernelArg(ocl_kernel_spts_analyser, 3, sizeof(cl_mem), (void*)&d_csrRowHisto);

    // warmup device
    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        // memset d_csrRowHisto to 0
        err = clEnqueueWriteBuffer(ocl_command_queue, d_csrRowHisto, CL_TRUE, 0, m  * sizeof(int), csrRowHisto, 0, NULL, NULL);
        if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

        err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_spts_analyser, 1,
                                     NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
        if(err != CL_SUCCESS) { printf("ocl_kernel_spts_analyser kernel run error = %i\n", err); return err; }

        err = clWaitForEvents(1, &ceTimer);
        if(err != CL_SUCCESS) { printf("event error = %i\n", err); return err; }
    }

    double time_opencl_analysis = 0;
    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        // memset d_csrRowHisto to 0
        err = clEnqueueWriteBuffer(ocl_command_queue, d_csrRowHisto, CL_TRUE, 0, m  * sizeof(int), csrRowHisto, 0, NULL, NULL);
        if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

        err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_spts_analyser, 1,
                                     NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
        if(err != CL_SUCCESS) { printf("ocl_kernel_spts_analyser kernel run error = %i\n", err); return err; }

        err = clWaitForEvents(1, &ceTimer);
        if(err != CL_SUCCESS) { printf("event error = %i\n", err); return err; }

        basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
        time_opencl_analysis += double(endTime - startTime) / 1000000.0;
    }

    time_opencl_analysis /= BENCH_REPEAT;

    printf("opencl syncfree SpTS analysis on L used %4.2f ms\n", time_opencl_analysis);

    // validate csrRowHisto
    err = clEnqueueReadBuffer(ocl_command_queue, d_csrRowHisto, CL_TRUE, 0, m * sizeof(int), csrRowHisto, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    int err_counter = 0;
    for (int i = 0; i < m; i++)
    {
        //printf("[%i]: csrRowPtrL = %i, csrRowPtrL_tmp = %i\n", i, csrRowPtrL[i], csrRowPtrL_tmp[i]);
        if (csrRowHisto[i] != csrRowPtrL_tmp[i+1] - csrRowPtrL_tmp[i])
            err_counter++;
    }

    if (!err_counter)
        printf("opencl syncfree SpTS analyser on L passed!\n");
    else
        printf("opencl syncfree SpTS analyser on L failed!\n");

    //  - opencl syncfree SpTS solve start!
    printf(" - opencl syncfree SpTS solve start!\n");

    // step 5: solve L*y = x
    const int wpb = WARP_PER_BLOCK;

    num_threads = WARP_PER_BLOCK * WARP_SIZE;
    num_blocks = ceil ((double)m / (double)(num_threads/WARP_SIZE));

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    err  = clSetKernelArg(ocl_kernel_spts_executor, 0,  sizeof(cl_mem), (void*)&d_cscColPtrL);
    err |= clSetKernelArg(ocl_kernel_spts_executor, 1,  sizeof(cl_mem), (void*)&d_cscRowIdxL);
    err |= clSetKernelArg(ocl_kernel_spts_executor, 2,  sizeof(cl_mem), (void*)&d_cscValL);
    err |= clSetKernelArg(ocl_kernel_spts_executor, 3,  sizeof(cl_mem), (void*)&d_csrRowHisto);
    err |= clSetKernelArg(ocl_kernel_spts_executor, 4,  sizeof(cl_mem), (void*)&d_left_sum);
    err |= clSetKernelArg(ocl_kernel_spts_executor, 5,  sizeof(cl_int), (void*)&m);
    err |= clSetKernelArg(ocl_kernel_spts_executor, 6,  sizeof(cl_int), (void*)&nnzL);
    err |= clSetKernelArg(ocl_kernel_spts_executor, 7,  sizeof(cl_mem), (void*)&d_b);
    err |= clSetKernelArg(ocl_kernel_spts_executor, 8,  sizeof(cl_mem), (void*)&d_x);
    err |= clSetKernelArg(ocl_kernel_spts_executor, 9,  sizeof(cl_int) * WARP_PER_BLOCK, NULL);
    err |= clSetKernelArg(ocl_kernel_spts_executor, 10, sizeof(VALUE_TYPE) * WARP_PER_BLOCK, NULL);
    err |= clSetKernelArg(ocl_kernel_spts_executor, 11, sizeof(cl_int), (void*)&wpb);

    double time_opencl_solve = 0;
    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        // set d_csrRowHisto to initial values
        err = clEnqueueWriteBuffer(ocl_command_queue, d_csrRowHisto, CL_TRUE, 0, m  * sizeof(int), csrRowHisto, 0, NULL, NULL);
        if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

        // memset d_left_sum to 0
        err = clEnqueueWriteBuffer(ocl_command_queue, d_left_sum, CL_TRUE, 0, m  * sizeof(VALUE_TYPE), left_sum, 0, NULL, NULL);
        if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

        err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_spts_executor, 1,
                                     NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
        if(err != CL_SUCCESS) { printf("ocl_kernel_spts_analyser kernel run error = %i\n", err); return err; }

        err = clWaitForEvents(1, &ceTimer);
        if(err != CL_SUCCESS) { printf("event error = %i\n", err); return err; }

        basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
        time_opencl_solve += double(endTime - startTime) / 1000000.0;
    }

    time_opencl_solve /= BENCH_REPEAT;

    printf("opencl syncfree SpTS solve used %4.2f ms, throughput is %4.2f gflops\n",
           time_opencl_solve, 2*nnzL/(1e6*time_opencl_solve));

    err = clEnqueueReadBuffer(ocl_command_queue, d_x, CL_TRUE, 0, n * sizeof(VALUE_TYPE), x, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // validate x
    err_counter = 0;
    for (int i = 0; i < n; i++)
    {
        if (abs(x_ref[i] - x[i]) > 0.01 * abs(x_ref[i]))
            err_counter++;
    }

    if (!err_counter)
        printf("opencl syncfree SpTS on L passed!\n");
    else
        printf("opencl syncfree SpTS on L failed!\n");

    // step 6: free resources
    free(csrRowHisto);

    if(d_csrRowHisto) err = clReleaseMemObject(d_csrRowHisto); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_left_sum) err = clReleaseMemObject(d_left_sum); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    if(d_cscColPtrL) err = clReleaseMemObject(d_cscColPtrL); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_cscRowIdxL) err = clReleaseMemObject(d_cscRowIdxL); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_cscValL) err = clReleaseMemObject(d_cscValL); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_b) err = clReleaseMemObject(d_b); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_x) err = clReleaseMemObject(d_x); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    return 0;
}

#endif



