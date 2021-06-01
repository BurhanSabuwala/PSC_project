# Submission folder for ID5130 Parallel Scientific Computing course project
Authors: Burhanuddin Sabuwala (BE17B011), Prashant G (BS17B011)

## Serial code
File: serial_code_final.c
commands to run the code:
```
gcc -fopenmp serial_code_final.c -lm
./a.out 50 50 50 25 0.18 80 10000 1e-8
./a.out 100 100 100 50 0.14 120 100000 1e-8
./a.out 200 200 200 100 0.105 180 100000 1e-4
```
Note: openmp used during compilation is only for profiling the code.
```
./a.out n1 n2 n3 F congruence cp max_iter tol
```
- n1, n2, n3 are int data types. They specify the dimensions of the tensor. In our experiments we considered all three of them to be equal.
- (int) F is the true rank of the tensor. 
- (double) congruence is used to adjust the colinearity. Colinearity decides how 
- (int) cp is the number of components present in factor matrix obtained during decomposition
- (int) max_iter is the maximum number of iterations
- (double) tolerance in the error between factor matrices obtained in two consecutive iterations

## OpenMP code
File: openmp_code_final.c
commands to run the code:
```
gcc -fopenmp openmp_code_final.c -lm
./a.out 50 50 50 25 0.18 80 10000 1e-8 2
./a.out 100 100 100 50 0.14 120 100000 1e-8 4
./a.out 200 200 200 100 0.105 180 100000 1e-4 32
```
```
./a.out n1 n2 n3 F congruence cp max_iter tol num_threads
```
- n1, n2, n3 are int data types. They specify the dimensions of the tensor. In our experiments we considered all three of them to be equal.
- (int) F is the true rank of the tensor. 
- (double) congruence is used to adjust the colinearity. Colinearity decides how 
- (int) cp is the number of components present in factor matrix obtained during decomposition
- (int) max_iter is the maximum number of iterations
- (double) tolerance in the error between factor matrices obtained in two consecutive iterations
- (int) num_threads is the number of threads for parallelizing the code

## OpenACC code
File: openacc_code_final.c
commands to run the code:
```
module load nvhpc-compiler
/sware/hpc_sdk/Linux_x86_64/2020/compilers/bin/pgcc -acc -Minfo=accel $PBS_O_WORKDIR/openacc_code_final.c >>output 2>>output
./a.out 50 50 50 25 0.18 80 10000 1e-8 50
./a.out 100 100 100 50 0.14 120 100000 1e-8 100
./a.out 200 200 200 100 0.105 180 100000 1e-4 200
```
Note: openmp used during compilation is only for profiling the code.
```
./a.out n1 n2 n3 F congruence cp max_iter tol num_gangs
```
- n1, n2, n3 are int data types. They specify the dimensions of the tensor. In our experiments we considered all three of them to be equal.
- (int) F is the true rank of the tensor. 
- (double) congruence is used to adjust the colinearity. Colinearity decides how 
- (int) cp is the number of components present in factor matrix obtained during decomposition
- (int) max_iter is the maximum number of iterations
- (double) tolerance in the error between factor matrices obtained in two consecutive iterations
- (int) num_gangs number of gangs specified to be used for parallelizing the task using openacc
