#!/bin/bash
for i in {1..4}
do
	srun -p gpudev -n 1 -t 00:10:00 ./Debug/MxMT.out8 < ./log/inputData0.prn >> ./log/r0.csv
done
