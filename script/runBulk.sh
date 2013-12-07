#!/bin/bash
for i in {1..4}
do
	for j in {4,8,16,24,32}
	do
		srun -p gpudev -n 1 -t 00:10:00 ./Debug/MxMT.out$j < ./log/inputData1.prn >> ./log/r1.csv
	done
done