#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda.h>
#include "MxMTconst.h"
#include "cudaMxMT.cuh"

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

//diagonal
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
		if (srcRowC < d && srcCol < d )
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
