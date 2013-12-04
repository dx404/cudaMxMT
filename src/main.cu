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
	if (argc < 3){
		cerr << "usage: " << "<program> <matrix dimension> <version>" << endl;
		return -1;
	}
	int dim = atoi(argv[1]);
	int version = atoi(argv[2]);
	float timeElapsed = 0;
	float gFlops;

	cudaGFlopTimer *cgTimer = new cudaGFlopTimer();
	dim3 gDim(ceil((float) dim / BLOCK_WIDTH ), ceil((float)dim/BLOCK_WIDTH), 1);
	dim3 bDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

	float *m = new float [dim*dim];
	float *mr = new float [dim*dim]; //cuda result
	float *mp = new float [dim*dim]; //processor comparison results
	matrixPopulate(m, dim, false);
	float *d_mr, *d_m;
	cudaMalloc((void **) &d_mr, dim*dim*sizeof(float));
	cudaMalloc((void **) &d_m,  dim*dim*sizeof(float));
	cudaMemcpy(d_m, m, dim*dim*sizeof(float), cudaMemcpyHostToDevice);

	cgTimer->start();
	switch(version){
	case 0:
		cout << "cu BLAS" << endl;
		cuBLAS_MxMT_device(d_mr, d_m, dim);
		break;
	case 1:
		cuda_MxMT_v001<<< gDim, bDim >>>(d_mr, d_m, dim);
		break;
	case 2:
		cuda_MxMT_v002<<< gDim, bDim >>>(d_mr, d_m, dim);
		break;
	case 3:
		cuda_MxMT_v003<<< gDim, bDim >>>(d_mr, d_m, dim);
		break;
	case 4:
		cuda_MxMT_v004<<< gDim, bDim >>>(d_mr, d_m, dim);
		break;
	default:
		cuBLAS_MxMT_device(d_mr, d_m, dim);
		break;
	}
	cgTimer->stop();

	cudaMemcpy(mr, d_mr, dim*dim*sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	seq_MxMT(mp, m, dim);
	if (dim < 256){
		diplayMatrix(m, dim);
		diplayMatrix(mr, dim);
	}

	gFlops = cgTimer->getGFlops(dim);
	timeElapsed = cgTimer->getElapsedTime();

	cout << "L2 diff: " << MatrixL2Diff(mp, mr, dim) << endl;
	cout << "Elapsed time: " << timeElapsed << endl;
	cout << "GPIS: " <<  gFlops << endl;
	delete[] m;
	delete[] mr;
	delete[] mp;
	delete cgTimer;
	cudaFree(d_mr);
	cudaFree(d_m);
	return 0;
}
