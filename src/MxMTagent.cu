#include <iostream>
#include <cuda.h>
#include "MxMTconst.h"
#include "cuBLAS_MxMT.cuh"
#include "cudaMxMT.cuh"
#include "seqMatrix.h"
#include "MxMTagent.cuh"
using namespace std;

cudaMxMT_agent::cudaMxMT_agent(int dim) : BW(BLOCK_WIDTH) {
	cgTimer = new cudaGFlopTimer();
	cgTimerCopy = new cudaGFlopTimer();
	this->dim = dim;
	this->byteSize = dim*dim*sizeof(float);
	this->san_check = false;
	this->disp_result = false;

	hostSrcMatrix = new float [dim*dim];
	hostTargetMatrix = new float [dim*dim];
	hostCheckMatrix = new float [dim*dim];

	cudaMalloc((void **) &deviceSrcMatrix, byteSize);
	cudaMalloc((void **) &deviceTargetMatrix, byteSize);

	gDim.x = dim / BLOCK_WIDTH + 1;
	gDim.y = dim / BLOCK_WIDTH + 1;
	gDim.z = 1;

	bDim.x = BLOCK_WIDTH;
	bDim.y = BLOCK_WIDTH;
	bDim.z = 1;

}

cudaMxMT_agent::~cudaMxMT_agent(){
	delete[] hostSrcMatrix;
	delete[] hostTargetMatrix;
	delete[] hostCheckMatrix;
	delete cgTimer;
	delete cgTimerCopy;
	cudaFree(deviceSrcMatrix);
	cudaFree(deviceTargetMatrix);
}

void cudaMxMT_agent::cudaMxMT_init_rand(int init_method){
	this->init_method = init_method;
	matrixPopulate(hostSrcMatrix, dim, init_method);
	matrixPopulate(hostTargetMatrix, dim, 2);
}

void cudaMxMT_agent::cudaMxMT_calculate(int impl_version){
	this->impl_version = impl_version;
	cgTimerCopy->start();
	cudaMemcpy(deviceSrcMatrix, hostSrcMatrix, byteSize, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceTargetMatrix, hostTargetMatrix, byteSize, cudaMemcpyHostToDevice);
	cgTimer->start();
	switch((int)impl_version){
	case 0:
		cuBLAS_MxMT_device(deviceTargetMatrix, deviceSrcMatrix, dim);
		break;
	case 1:
		cuda_MxMT_v001<<< gDim, bDim >>>(deviceTargetMatrix, deviceSrcMatrix, dim);
		break;
	case 2:
		cuda_MxMT_v002<<< gDim, bDim >>>(deviceTargetMatrix, deviceSrcMatrix, dim);
		break;
	case 3:
		cuda_MxMT_v003<<< gDim, bDim >>>(deviceTargetMatrix, deviceSrcMatrix, dim);
		break;
	case 4:
		cuda_MxMT_v004<<< gDim, bDim >>>(deviceTargetMatrix, deviceSrcMatrix, dim);
		break;
	case 5:
		cuda_MxMT_v005<<< gDim, bDim >>>(deviceTargetMatrix, deviceSrcMatrix, dim);
		break;
	case 6:
		cuda_MxMT_v006<<< gDim, bDim >>>(deviceTargetMatrix, deviceSrcMatrix, dim);
		break;
	case 7:
		cuda_MxMT_v007<<< gDim, bDim >>>(deviceTargetMatrix, deviceSrcMatrix, dim);
		break;
	case 8:
		cuda_MxMT_v008<<< dim3(1, dim/BLOCK_WIDTH+1, 1), dim3(1024/BLOCK_WIDTH, BLOCK_WIDTH, 1) >>>(deviceTargetMatrix, deviceSrcMatrix, dim);
		break;
	case 9:
		cuda_MxMT_v009<<< dim3(1, dim/BLOCK_WIDTH+1, 1), dim3(1024/BLOCK_WIDTH, BLOCK_WIDTH, 1) >>>(deviceTargetMatrix, deviceSrcMatrix, dim);
		break;
	default:
		cuBLAS_MxMT_device(deviceTargetMatrix, deviceSrcMatrix, dim);
		break;
	}
	cgTimer->stop();
	cudaMemcpy(hostTargetMatrix, deviceTargetMatrix, byteSize, cudaMemcpyDeviceToHost);
	cgTimerCopy->stop();
	cudaDeviceSynchronize();
}

void cudaMxMT_agent::check(){
	this->san_check = true;
	seq_MxMT(hostCheckMatrix, hostSrcMatrix, dim);
	L2_diff = MatrixL2Diff(hostCheckMatrix, hostTargetMatrix, dim);
}

void cudaMxMT_agent::display_Matrix(){
	this->disp_result = true;
	cout << "Source Matrix" << endl;
	diplayMatrix(hostSrcMatrix, dim);
	cout << "Target Matrix from GPU" << endl;
	diplayMatrix(hostTargetMatrix, dim);
	if (san_check){
		cout << "Target Matrix from CPU" << endl;
		diplayMatrix(hostCheckMatrix, dim);
	}
}

void cudaMxMT_agent::printPofile(bool isCSV = true){
	float timeElapsed = cgTimer->getElapsedTime();
	float gFlops = cgTimer->getGFlops(dim);
	float timeElapsedCopy = cgTimerCopy->getElapsedTime();
	float gFlopsCopy= cgTimerCopy->getGFlops(dim);

	if (isCSV){
		cout << impl_version << ", " ;
		cout << dim << ", ";
		cout << BLOCK_WIDTH << ", ";
		cout << timeElapsed << ", ";
		cout << gFlops << ", ";
		cout << timeElapsedCopy << ", ";
		cout << gFlopsCopy << ", ";
		cout << san_check << ", ";
		cout << L2_diff << endl;
	}
	else{
		cout << "impl_version: " << impl_version << endl ;
		cout << "dim: " << dim << endl;
		cout << "BLOCK_WIDTH: " << BLOCK_WIDTH << endl;
		cout << "timeElapsed: " << timeElapsed << endl;
		cout << "gFlops: " << gFlops << endl;
		cout << "timeElapsedCopy: " << timeElapsedCopy << endl;
		cout << "gFlopsCopy: " << gFlopsCopy << endl;
		cout << "san_check: " << san_check << endl;
		cout << "L2_diff: " << L2_diff << endl;
		cout << endl;
	}
}
