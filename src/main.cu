#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include "seqMatrix.h"
#include "MxMTconst.h"
#include "cudaMxMT.cuh"
#include "cuBLAS_MxMT.cuh"
using namespace std;

/**
 * Author: Duo Donald Zhao
 */

int main(int argc, char *argv[]){
	cout << "Hello, Welcome to the world of CUDA Matrix Multiplication!" << endl;
	if (argc < 2){
		cerr << "usage: " << "<program> <matrix dimension>" << endl;
		return -1;
	}
	int dim = atoi(argv[1]);
	cout << "Starting Computing Matrix Multiplication of dimension: " << dim << endl;
	cout << "Initializting Matrix" << endl;
	float *m = new float [dim*dim];
	float *mr = new float [dim*dim]; //cuda result
	float *mp = new float [dim*dim]; //processor comparison results
	matrixPopulate(m, dim, false);
	//copy(m, m + dim*dim, mr); //STL copying
	//diplayMatrix(m, dim);
	//	diplayMatrix(mr, dim);
	seq_MxMT(mp, m, dim);
	//diplayMatrix(mp, dim);

	cout << "Copying Data To GPU" << endl;
	float *d_mr, *d_m;
	cudaMalloc((void **) &d_mr, dim*dim*sizeof(float));
	cudaMalloc((void **) &d_m, dim *dim*sizeof(float));
	cudaMemcpy(d_m, m, dim*dim*sizeof(float), cudaMemcpyHostToDevice);

	cout << "Invoking GPU kernel to Compute" << endl;
	dim3 gDim(ceil((float) dim / BLOCK_WIDTH ), ceil((float)dim/BLOCK_WIDTH), 1);
	//	dim3 gDim(128, 128, 1);
	dim3 bDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	/**
	 * 1024 512 256 32
	 */
	// cuda_MxMT_naive<<< dim3(dim/256+1,dim/256+1,1), dim3(1024,1024,1) >>>(d_mr, d_m, dim);
	cuda_MxMT_v001<<< gDim, bDim >>>(d_mr, d_m, dim);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaDeviceSynchronize();
	cout << "Copying Data back from GPU kernel" << endl;
	cudaMemcpy(mr, d_mr, dim*dim*sizeof(float), cudaMemcpyDeviceToHost);

	cout << "Displaying Results: " << endl;
	//diplayMatrix(mr, dim);
	cudaDeviceSynchronize();

	cout << "L2 diff: " << MatrixL2Diff(mp, mr, dim) << endl;
	cout << "Elapsed time: " << time << endl;
	cout << "GPIS: " <<  ( 2.0e-6 * dim * dim * dim) /(time) << endl;
	delete[] m;
	delete[] mr;
	delete[] mp;
	cudaFree(d_mr);
	cudaFree(d_m);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cout << "prop.maxThreadsPerBlock: " << prop.maxThreadsPerBlock << endl;

	return 0;
}
