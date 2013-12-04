#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda.h>
using namespace std;

const int BLOCK_WIDTH = 8;

#if BWX32
	const BLOCK_WX = 32;
#elif BWX24
	const BLOCK_WX = 24;
#elif BWX16
	const BLOCK_WX = 16;
#elif BWX8
	const BLOCK_WX = 8;
#else
	const BLOCK_WX = 4;
#endif

#if BWY32
	const BLOCK_WY = 32;
#elif BWY24
	const BLOCK_WY = 24;
#elif BWY16
	const BLOCK_WY = 16;
#elif BWY8
	const BLOCK_WY = 8;
#else
	const BLOCK_WY = 4;
#endif


/**
 * Author: Duo Donald Zhao
 */
// Sequential Matrix Multiplication for computing M * M'
void seq_MxMT(float *mr, float *m, int d){
	for (int i = 0; i < d; i++){
		for (int j = 0; j < d; j++){
			float sum = 0;
			for (int k = 0; k < d; k++){
				sum += m[i*d+k] * m[j*d+k];
			}
			mr[i*d+j] = sum;
		}
	}
}

__global__ void cuda_MxMT_naive(float *d_mr, float *d_m, int d){
	int bdx = blockIdx.x, bdy = blockIdx.y;
	int tdx = threadIdx.x, tdy = threadIdx.y;
	int j = bdx * blockDim.x + tdx;
	int i = bdy * blockDim.y + tdy;
	if (i < d && j < d){
		float sum = 0;
		for (int k = 0; k < d; k++){
			sum += d_m[i*d+k] * d_m[j*d+k];
		}
		d_mr[i*d+j] = sum;
	}
}

//power 2
__global__ void cuda_MxMT_v001 (float *d_mr, float *d_m, int d){
	float rval = 0;
	__shared__ float b_mr[BLOCK_WIDTH][BLOCK_WIDTH];
	__shared__ float b_mc[BLOCK_WIDTH][BLOCK_WIDTH];

	int bdx = blockIdx.x, bdy = blockIdx.y;
	int tdx = threadIdx.x, tdy = threadIdx.y;
	int row = bdx * blockDim.x + tdx;
	int col = bdy * blockDim.y + tdy;
	int rpos = row * d + col;

	for (int bk = 0; bk < gridDim.y; bk++){
		b_mr[tdx][tdy] = d_m[(bdx*BLOCK_WIDTH + tdx)*d + bk * BLOCK_WIDTH + tdy]; // (bdx*w+tdx, bk*w+tdy)
		b_mc[tdx][tdy] = d_m[(bdy*BLOCK_WIDTH + tdx)*d + bk * BLOCK_WIDTH + tdy];	// (bdy*w+tdx, bk*w+tdy)
		__syncthreads();
		for (int tk = 0; tk < BLOCK_WIDTH; tk++){
			rval +=  b_mr[tdx][tk] * b_mc[tdy][tk];
		}
		__syncthreads();
	}
	d_mr[rpos] = rval;
}

//general
__global__ void cuda_MxMT_v002 (float *d_mr, float *d_m, int d){
	float rval = 0;
	__shared__ float b_mr[BLOCK_WIDTH][BLOCK_WIDTH];
	__shared__ float b_mc[BLOCK_WIDTH][BLOCK_WIDTH];

	int bdx = blockIdx.x, bdy = blockIdx.y;
	int tdx = threadIdx.x, tdy = threadIdx.y;
	int row = bdx * blockDim.x + tdx;
	int col = bdy * blockDim.y + tdy;
	int rpos = row * d + col;

	int srcRowR = bdx*BLOCK_WIDTH + tdx;
	int srcRowC = bdy*BLOCK_WIDTH + tdx;
	for (int bk = 0; bk < gridDim.y; bk++){
		int srcCol = bk * BLOCK_WIDTH + tdy;
		if (srcRowR < d && srcCol < d)
			b_mr[tdx][tdy] = d_m[srcRowR*d + srcCol]; // (bdx*w+tdx, bk*w+tdy)
		if (srcRowC < d && srcCol < d)
			b_mc[tdx][tdy] = d_m[srcRowC*d + srcCol];	// (bdy*w+tdx, bk*w+tdy)

		__syncthreads();
		if (row < d && col < d){
			for (int tk = 0; tk < BLOCK_WIDTH; tk++){
				if (bk*BLOCK_WIDTH + tk < d)
					rval +=  b_mr[tdx][tk] * b_mc[tdy][tk];
			}
		}
		__syncthreads();
	}
	if (row < d && col < d){
		d_mr[rpos] =  rval;// + 0.1*row + 0.01*col;
	}
}

__global__ void cuda_MxMT_v003 (float *d_mr, float *d_m, int d){
	float rval = 0;
	__shared__ float b_mr[BLOCK_WIDTH][BLOCK_WIDTH];
	__shared__ float b_mc[BLOCK_WIDTH][BLOCK_WIDTH];

	int bdx = blockIdx.x, bdy = blockIdx.y;
	int tdx = threadIdx.x, tdy = threadIdx.y;
	int row = bdx * blockDim.x + tdx;
	int col = bdy * blockDim.y + tdy;
	int rpos = row * d + col;

	int srcRowR = bdx*BLOCK_WIDTH + tdx;
	int srcRowC = bdy*BLOCK_WIDTH + tdx;
	for (int bk = 0; bk < gridDim.y; ++bk){
		int srcCol = bk * BLOCK_WIDTH + tdy;
		if (srcRowR < d && srcCol < d)
			b_mr[tdx][tdy] = d_m[srcRowR*d + srcCol]; // (bdx*w+tdx, bk*w+tdy)
		if (srcRowC < d && srcCol < d)
			b_mc[tdx][tdy] = d_m[srcRowC*d + srcCol];	// (bdy*w+tdx, bk*w+tdy)

		__syncthreads();
		if (row < d && col < d){
			if (bk != gridDim.y - 1)
				for (int tk = 0; tk < BLOCK_WIDTH; tk++)
					rval +=  b_mr[tdx][tk] * b_mc[tdy][tk];
			else
				for (int tk = 0; tk < d % BLOCK_WIDTH; tk++)
					rval +=  b_mr[tdx][tk] * b_mc[tdy][tk];
		}
		__syncthreads();
	}
	if (row < d && col < d){
		d_mr[rpos] =  rval;// + 0.1*row + 0.01*col;
	}
}

__global__ void cuda_MxMT_v004 (float *d_mr, float *d_m, int d){
	float rval = 0;
	__shared__ float b_mr[BLOCK_WIDTH][BLOCK_WIDTH];
	__shared__ float b_mc[BLOCK_WIDTH][BLOCK_WIDTH];

	int bdx = blockIdx.x, bdy = blockIdx.y;
	int tdx = threadIdx.x, tdy = threadIdx.y;
	if (bdx > bdy)
		return;
	int row = bdx * blockDim.x + tdx;
	int col = bdy * blockDim.y + tdy;
	int rpos = row * d + col;
	int rposPair = col * d + row;
	int srcRowR = bdx*BLOCK_WIDTH + tdx;
	int srcRowC = bdy*BLOCK_WIDTH + tdx;
	for (int bk = 0; bk < gridDim.y; ++bk){
		int srcCol = bk * BLOCK_WIDTH + tdy;
		if (srcRowR < d && srcCol < d)
			b_mr[tdx][tdy] = d_m[srcRowR*d + srcCol]; // (bdx*w+tdx, bk*w+tdy)
		if (srcRowC < d && srcCol < d)
			b_mc[tdx][tdy] = d_m[srcRowC*d + srcCol];	// (bdy*w+tdx, bk*w+tdy)

		__syncthreads();
		if (row < d && col < d){
			if (bk != gridDim.y - 1)
				for (int tk = 0; tk < BLOCK_WIDTH; tk++)
					rval +=  b_mr[tdx][tk] * b_mc[tdy][tk];
			else
				for (int tk = 0; tk < d % BLOCK_WIDTH; tk++)
					rval +=  b_mr[tdx][tk] * b_mc[tdy][tk];
		}
		__syncthreads();
	}
	if (row < d && col < d){
		d_mr[rposPair] = d_mr[rpos] =  rval;// + 0.1*row + 0.01*col;
	}
}

__global__ void cuda_MxMT_v005 (float *d_mr, float *d_m, int d){
	float rval = 0;
	__shared__ float b_mr[BLOCK_WIDTH][BLOCK_WIDTH];
	__shared__ float b_mc[BLOCK_WIDTH][BLOCK_WIDTH];

	int bdx = blockIdx.x, bdy = blockIdx.y;
	int tdx = threadIdx.x, tdy = threadIdx.y;
	if (bdx > bdy)
		return;
	int row = bdx * blockDim.x + tdx;
	int col = bdy * blockDim.y + tdy;
	int rpos = row * d + col;
	int rposPair = col * d + row;
	int srcRowR = bdx*BLOCK_WIDTH + tdx;
	int srcRowC = bdy*BLOCK_WIDTH + tdx;
	for (int bk = 0; bk < gridDim.y; ++bk){
		int srcCol = bk * BLOCK_WIDTH + tdy;
		if (srcRowR < d && srcCol < d)
			b_mr[tdx][tdy] = d_m[srcRowR*d + srcCol]; // (bdx*w+tdx, bk*w+tdy)
//		if (srcRowC < d && srcCol < d)
//			b_mc[tdx][tdy] = d_m[srcRowC*d + srcCol];	// (bdy*w+tdx, bk*w+tdy)

		__syncthreads();
		if (row < d && col < d){
//			if (bk != gridDim.y - 1){
				rval =  b_mr[tdx][0] * b_mc[tdy][0]
					+ b_mr[tdx][1] * b_mc[tdy][1];
//					+ b_mr[tdx][2] * b_mc[tdy][2]
//					+ b_mr[tdx][3] * b_mc[tdy][3];
//					+ b_mr[tdx][4] * b_mc[tdy][4]
//					+ b_mr[tdx][5] * b_mc[tdy][5]
//					+ b_mr[tdx][6] * b_mc[tdy][6]
//					+ b_mr[tdx][7] * b_mc[tdy][7];
//			}
//			else
//				for (int tk = 0; tk < d % BLOCK_WIDTH; tk++)
//					rval +=  b_mr[tdx][tk] * b_mc[tdy][tk];
		}
		__syncthreads();
	}
	if (row < d && col < d){
		d_mr[rposPair] = d_mr[rpos] =  rval;// + 0.1*row + 0.01*col;
	}
}


// for CUDA consistency all matrix are represented by 1D
//Random Square Matrix Populator
void matrixPopulate(float *m, int d, bool fromRand = true){
	if (fromRand){
		srand(time(0));
		for (int i = 0; i < d*d; i++){
			m[i] = rand()/(float) RAND_MAX;
		}
	}
	else{
		for (int i = 0; i < d*d; i++){
			m[i] = i % 1000;
		}
	}
}

// Matrix Display
void diplayMatrix(float *m, int d){
	cout << "m={" << endl;
	for (int i = 0; i < d; i++){
		cout << "\t{";
		for (int j = 0; j < d - 1; j++){
			cout << setw(10) << m[i*d+j] << ", ";
		}
		cout << setw(10) << m[i*d + d - 1] <<
				((i==d-1)? "}" : "},") << endl;
	}
	cout << "};" << endl;
}

//Matrix Verifier
float MatrixL2Diff(float *mt, float *ms, int d){
	float sum = 0;
	for (int i = 0; i < d * d; i++){
		float diff = mt[i] - ms[i];
		sum += diff * diff;
	}
	return sum;
}

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
	dim3 bDim(BLOCK_WIDTH,BLOCK_WIDTH,1);

	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	/**
	 * 1024 512 256 32
	 */
	// cuda_MxMT_naive<<< dim3(dim/256+1,dim/256+1,1), dim3(1024,1024,1) >>>(d_mr, d_m, dim);
	cuda_MxMT_v005<<< gDim, bDim >>>(d_mr, d_m, dim);

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

	int count;
	cudaGetDeviceCount(&count);
	cout << count << endl;
	return 0;
}
