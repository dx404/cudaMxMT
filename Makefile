PROJECT_DIR=$(HOME)/MxMT/MxMT
CC=nvcc
OPTIMIZATION=-O1
cudaVersion=-arch=compute_30 -code=sm_30
srcPath=$(PROJECT_DIR)/src/main.cu
exePath=$(PROJECT_DIR)/Debug/MxMT.out
#oFLAG=-ptxas-options=-v -maxrregcount 60

all: MxMT.out
	
MxMT.out: main.o seqMatrix.o cudaMxMT.o cuBLAS_MxMT.o
	nvcc $(cudaVersion) $(OPTIMIZATION) $(oFLAG) \
	$(PROJECT_DIR)/Debug/main.o \
	$(PROJECT_DIR)/Debug/seqMatrix.o \
	$(PROJECT_DIR)/Debug/cudaMxMT.o \
	$(PROJECT_DIR)/Debug/cuBLAS_MxMT.o \
	-o $(PROJECT_DIR)/Debug/MxMT.out -lcublas

main.o: $(PROJECT_DIR)/src/main.cu
	nvcc -c $(cudaVersion) $(OPTIMIZATION) $(oFLAG) \
	$(PROJECT_DIR)/src/main.cu \
	-o $(PROJECT_DIR)/Debug/main.o
	
seqMatrix.o: $(PROJECT_DIR)/src/seqMatrix.cpp
	g++ -c $(PROJECT_DIR)/src/seqMatrix.cpp \
	-o $(PROJECT_DIR)/Debug/seqMatrix.o

cudaMxMT.o: $(PROJECT_DIR)/src/cudaMxMT.cu
	nvcc -c $(cudaVersion) $(OPTIMIZATION) $(oFLAG) \
	$(PROJECT_DIR)/src/cudaMxMT.cu \
	-o $(PROJECT_DIR)/Debug/cudaMxMT.o

cuBLAS_MxMT.o: seqMatrix.o
	nvcc -c $(cudaVersion) $(OPTIMIZATION) $(oFLAG) \
	$(PROJECT_DIR)/src/cuBLAS_MxMT.cu \
	-o $(PROJECT_DIR)/Debug/cuBLAS_MxMT.o -lcublas

clean:
	rm -fr Debug/*.o Debug/*.out
	rm -fr src/*.o src/*.out
	rm -fr *.o *.out
	
run:
	srun -p gpudev -n 1 -t 00:10:00 ./Debug/outMxMT.out 2222
	
	