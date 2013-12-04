/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "seqMatrix.h"
#include "cuBLAS_MxMT.cuh"
using namespace std;

float cuBLAS_MxMT(float *h_m, float *h_r, int d){
	float time;

	float *d_m, *d_r;
	cudaMalloc((void **) &d_m, d*d*sizeof(float));
	cudaMalloc((void **) &d_r, d*d*sizeof(float));
	cudaMemcpy(d_m, h_m, d*d*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, h_r, d*d*sizeof(float), cudaMemcpyHostToDevice);
	cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0f, beta = 1.0f;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	cublasSgemm_v2(handle,
			CUBLAS_OP_T, CUBLAS_OP_N,
			d, d ,d,
			&alpha,
			d_m, d,
			d_m, d,
			&beta,
			d_r, d);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cudaMemcpy(h_r, d_r, d*d*sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_m);
	cudaFree(d_r);


	float gFLOPS =   ( 2.0e-6 * d * d * d) /(time);

	return gFLOPS;
}

/*
int main(int argc, char *argv[]) {
	int d = atoi(argv[1]);
	float *h_m = new float[d*d];
	float *h_r = new float[d*d];
	float *s_r = new float[d*d];
	matrixPopulate(h_m, d, true);
	matrixPopulate(h_r, d, 0.0f);
	//seq_MxMT(s_r, h_m, d);
	//diplayMatrix(h_m, d);
	//diplayMatrix(h_r, d);
	float gFLOPS = cuBLAS_MxMT(h_m, h_r, d);
	//diplayMatrix(h_r, d);
	cout << "gFlops = " << gFLOPS << endl;
	cout << "h_r[end] = " << h_r[d*d/2] << endl;
	//cout << "L2 diff:" << MatrixL2Diff(h_r, s_r, d) << endl;
	delete[] h_m;
	delete[] h_r;
	delete[] s_r;
	return 0;
}
*/
