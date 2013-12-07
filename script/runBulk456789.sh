#!/bin/bash
for v in {4,5,6,7}
do
	for i in {1..4}
	do
		for j in {4,8,16,24,32}
		do
			srun -p gpudev -n 1 -t 00:10:00 ./Debug/MxMT.out$j < ./log/inputData$v.prn >> ./log/r4_9.csv
		done
	done
done

for v in {8, 9}
do
	for i in {1..4}
	do
		for j in {8}
		do
			srun -p gpudev -n 1 -t 00:10:00 ./Debug/MxMT.out$j < ./log/inputData$v.prn >> ./log/r4_9.csv
		done
	done
done