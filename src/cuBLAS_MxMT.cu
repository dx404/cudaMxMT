/**
 * Author: Duo Zhao
 * This part is the
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "seqMatrix.h"
#include "cuBLAS_MxMT.cuh"
#include "cudaGFlopTimer.cuh"

void cuBLAS_MxMT_device(float *d_r, float *d_m, int d){
	cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0f, beta = 1.0f;
	// calling cuda buid-in library to executing computation
	cublasSgemm_v2(handle,
			CUBLAS_OP_T, CUBLAS_OP_N,
			d, d ,d,
			&alpha,
			d_m, d,
			d_m, d,
			&beta,
			d_r, d);
}

float cuBLAS_MxMT_host(float *h_r, float *h_m, int d){
	cudaGFlopTimer *tr = new cudaGFlopTimer();

	float *d_m, *d_r;
	cudaMalloc((void **) &d_m, d*d*sizeof(float));
	cudaMalloc((void **) &d_r, d*d*sizeof(float));
	cudaMemcpy(d_m, h_m, d*d*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, h_r, d*d*sizeof(float), cudaMemcpyHostToDevice);
	cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0f, beta = 1.0f;

	// calling cuda buid-in library to executing computation
	tr->start();
	cublasSgemm_v2(handle,
			CUBLAS_OP_T, CUBLAS_OP_N,
			d, d ,d,
			&alpha,
			d_m, d,
			d_m, d,
			&beta,
			d_r, d);
	tr->stop();

	cudaMemcpy(h_r, d_r, d*d*sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_m);
	cudaFree(d_r);
	float Gflops = tr->getGFlops(d);
	delete tr;
	return Gflops;
}

