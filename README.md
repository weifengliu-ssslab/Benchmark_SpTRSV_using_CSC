# Benchmark_SpTRSV_using_CSC
A Synchronization-Free Algorithm for Parallel Sparse Triangular Solves (SpTRSV)

<br><hr>
<h3>Introduction</h3>

This is the source code of the Euro-Par '16 paper "A Synchronization-Free Algorithm for Parallel Sparse Triangular Solves" by Weifeng Liu, Ang Li, Jonathan D. Hogg, Iain S. Duff, and Brian Vinter. [[PDF](http://www.nbi.dk/~weifeng/papers/sptrsv_liu_europar16.pdf)] [[Slides](http://www.nbi.dk/~weifeng/slides/sptrsv_liu_europar16_slides.pdf)] [[DOI](http://dx.doi.org/10.1007/978-3-319-43659-3_45)]

Update (14 Feb. 2020, cuda): A problem about deadlock on CUDA 10 has been fixed.

Update (13 Feb. 2017, cuda): A problem about caching has been fixed for Tesla P100. Thanks to Hartwig Anzt for identifying the probem and Ang Li for fixing it!

Update (30 Nov. 2016): This algorithm has been improved to support both forward and backward substitution, and multiple right-hand sides. See [https://github.com/bhSPARSE/Benchmark_SpTRSM_using_CSC](https://github.com/bhSPARSE/Benchmark_SpTRSM_using_CSC) for a newer version of this work.

Update (25 Feb. 2016, opencl): An OpenCL version has been added.

<br><hr>
<h3>nVidia GPU (CUDA) version</h3>

- Execution

1. Set CUDA path in the Makefile,
2. Run ``make``,
3. Run ``./sptrsv example.mtx``.

- Tested environments

1. nVidia GeForce GTX Titan X GPU in a host with CUDA v7.5 and Ubuntu 15.10 64-bit Linux installed.
2. nVidia Tesla K40c GPU in a host with CUDA v7.5 and Enterprise Linux installed. 
3. nVidia Geforce GT 650m GPU in a host with CUDA v7.5 and Mac OS X 10.9.2 installed.

- Data type

1. The code supports both double precision and single precision SpTRSV. Use ``make VALUE_TYPE=double`` for double precision or ``make VALUE_TYPE=float`` for single precision.

<br><hr>
<h3>AMD GPU (OpenCL 2.0) version</h3>

- Execution

1. Set OpenCL path in the Makefile,
2. Run ``make``,
3. Run ``./sptrsv example.mtx``.

- Tested environments (Note that an OpenCL 2.0 device is required for running the code)

1. AMD Radeon Fury X GPU in a host with AMD APP SDK 2.9.1 and Ubuntu 15.04 64-bit Linux installed.
2. AMD Radeon 290X GPU in a host with AMD APP SDK 2.9.1 and Ubuntu 15.04 64-bit Linux installed.

- Data type

1. The code supports both double precision and single precision SpTRSV. Use ``make VALUE_TYPE=double`` for double precision or ``make VALUE_TYPE=float`` for single precision. 

