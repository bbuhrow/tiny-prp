This is a program to benchmark fast routines for probable primality checking of small (< 104-bit) integers.
It uses AVX512 to test either 8 input integers in parallel or in the case of MR-sprp, 8 bases on one integer.

It includes tests of:
fermat-prp for < 52-bit integers (8 inputs in parallel)
fermat-prp for < 104-bit integers (8 inputs in parallel)
MR-sprp base 2 for < 52-bit integers (8 inputs in parallel)
MR-sprp base 2 for < 104-bit integers (8 inputs in parallel)
MR-sprp arbitrary base for < 104-bit integers (8 inputs in parallel)
MR-sprp arbitrary base for < 104-bit integers (8 bases in parallel)

Example build lines:

GCC using IFMA-equipped processor, running GMP checks (slow)
gcc -Ofast -march=icelake-client -DIFMA -g -m64 -std=gnu99 -I/path/to/gmp/include -L/path/to/gmp/lib -DGMP_CHECK prp.c -o prp -lgmp

GCC using IFMA-equipped processor
gcc -Ofast -march=icelake-client -DIFMA -g -m64 -std=gnu99 prp.c -o prp

GCC using non-IFMA-equipped processor
gcc -Ofast -march=skylake-avx512 -g -m64 -std=gnu99 prp.c -o prp

All of the above should also work using Intel's classic compiler - icc in place of gcc


