# Benchmark_SpTRSV_using_CSC
A Synchronization-Free Algorithm for Parallel Sparse Triangular Solves (SpTRSV)

<br><hr>
<h3>Introduction</h3>

This is the source code of the paper "A Synchronization-Free Algorithm for Parallel Sparse Triangular Solves" submitted to Euro-Par '16.

<br><hr>
<h3>nVidia GPU (CUDA) version</h3>

- Execution

1. Set CUDA path in the Makefile,
2. Run ``make``,
3. Run ``./sptrsv example.mtx``.

- Tested environments

1. nVidia GeForce GTX Titan X GPU in a host with Ubuntu 15.10 64-bit Linux installed.
2. nVidia Tesla K40c GPU in a host with Enterprise Linux installed.
3. nVidia Geforce GT 650m GPU in a host with Mac OS X 10.9.2 installed.

- Data type

1. The code supports both double precision and single precision SpTRSV. Use ``make VALUE_TYPE=double`` for double precision or ``make VALUE_TYPE=float`` for single precision.
