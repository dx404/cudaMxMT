/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "seqMatrix.h"
#include "cuBLAS_MxMT.cuh"

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
