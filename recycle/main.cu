#include <iostream>
#include "cudaGFlopTimer.cuh"
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

float cuBLAS_MxMT(float *h_m, float *h_r, int d){
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
	//timer->stop();

	cudaMemcpy(h_r, d_r, d*d*sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_m);
	cudaFree(d_r);
	float Gflops = 0; //= timer->getGFlops(d);
	return Gflops;
}




int main(int argc, char *argv[]){
  Hi *hi = new Hi();
  cudaGFlopTimer *cgt = new cudaGFlopTimer();
  cgt->start();
  cgt->stop();
  cgt->getElapsedTime();
  cgt->getGFlops(100);
  hi->x = 2;
  delete hi;
  delete cgt;
  return 0;
}
