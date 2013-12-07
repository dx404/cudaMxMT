#!/bin/bash
srun -p gpudev -n 1 -t 00:10:00 ./Debug/MxMT.out8 < ./log/inputDataSan.prn >> ./log/rSan.csv
