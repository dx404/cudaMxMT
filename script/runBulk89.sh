#!/bin/bash
for v in {8,9}
do
	for i in {1..4}
	do
		srun -p gpudev -n 1 -t 00:10:00 ./Debug/MxMT.out8 < ./log/inputData$v.prn >> ./log/r4_9.csv
	done
done