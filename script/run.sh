#!/bin/bash
#taking one argument as the dimension of the matrix
srun -p gpudev -n 1 -t 00:10:00 ./Debug/MxMT.out8 $1 $2 $3 $4 $5 $6