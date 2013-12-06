#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
#include "MxMTconst.h"
#include "cudaMxMT.cuh"

/*
 * The first version simply adopts the natural math computation order
 * which is directly translated from the sequential algorithm
 */
__global__ void cuda_MxMT_v001(float *d_mr, float *d_m, int d){
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

// multiple of block size
__global__ void cuda_MxMT_v002 (float *d_mr, float *d_m, int d){
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

	d_mr[rpos] =  rval;// + 0.1*row + 0.01*col;
}

//general
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

__global__ void cuda_MxMT_v004 (float *d_mr, float *d_m, int d){
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
__global__ void cuda_MxMT_v006 (float *d_mr, float *d_m, int d){
	float rval = 0;
	__shared__ float b_mr[BLOCK_WIDTH][BLOCK_WIDTH];
	__shared__ float b_mc[BLOCK_WIDTH][BLOCK_WIDTH];

	int bdx = blockIdx.x, bdy = blockIdx.y;
	int tdx = threadIdx.x, tdy = threadIdx.y;
	int row = bdx * blockDim.x + tdx;
	int col = bdy * blockDim.y + tdy;
	int rpos = row * d + col;
	int rposPair = col * d + row;
	int srcRowR = bdx*BLOCK_WIDTH + tdx;
	int srcRowC = bdy*BLOCK_WIDTH + tdx;
	if (bdx > bdy){
		return;
	}
	else if (bdx < bdy)
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
	else
		for (int bk = 0; bk < gridDim.y; ++bk){
			int srcCol = bk * BLOCK_WIDTH + tdy;
			if (srcRowR < d && srcCol < d)
				b_mr[tdx][tdy] = d_m[srcRowR*d + srcCol]; // (bdx*w+tdx, bk*w+tdy)

			__syncthreads();
			if (row < d && col < d){
				if (bk != gridDim.y - 1)
					for (int tk = 0; tk < BLOCK_WIDTH; tk++){
						rval +=   b_mr[tdx][tk] * b_mr[tdy][tk];
					}
				else
					for (int tk = 0; tk < d % BLOCK_WIDTH; tk++){
						rval +=  b_mr[tdx][tk] * b_mr[tdy][tk];
					}
			}
			__syncthreads();
		}

	if (row < d && col < d){
		d_mr[rposPair] = d_mr[rpos] =  rval;
	}
}

//loop unroll for BLOCK_WIDTH == 8
__global__ void cuda_MxMT_v007 (float *d_mr, float *d_m, int d){
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

	float rval = 0;
	for (int bk = 0; bk < gridDim.y; ++bk){
		int srcCol = bk * BLOCK_WIDTH + tdy;
		if (srcRowR < d && srcCol < d)
			b_mr[tdx][tdy] = d_m[srcRowR*d + srcCol]; // (bdx*w+tdx, bk*w+tdy)
		if (srcRowC < d && srcCol < d)
			b_mc[tdx][tdy] = d_m[srcRowC*d + srcCol];	// (bdy*w+tdx, bk*w+tdy)

		__syncthreads();
		if (row < d && col < d){
			if (bk != gridDim.y - 1){
				rval +=  b_mr[tdx][0] * b_mc[tdy][0];
				rval +=	 b_mr[tdx][1] * b_mc[tdy][1];
				rval +=	 b_mr[tdx][2] * b_mc[tdy][2];
				rval +=	 b_mr[tdx][3] * b_mc[tdy][3];
				rval +=	 b_mr[tdx][4] * b_mc[tdy][4];
				rval +=	 b_mr[tdx][5] * b_mc[tdy][5];
				rval +=	 b_mr[tdx][6] * b_mc[tdy][6];
				rval +=	 b_mr[tdx][7] * b_mc[tdy][7];
			}
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

// A single thread block
__global__ void cuda_MxMT_v008(float *d_mr, float *d_m, int d){
	__shared__ float bm[1024][BLOCK_WIDTH];
	int bdx = blockIdx.x, bdy = blockIdx.y;
	int tdx = threadIdx.x, tdy = threadIdx.y;
	int gtdx = bdx * blockDim.x + tdx;
	int gtdy = bdy * blockDim.y + tdy;

	for (int rowStart = 0; rowStart < d; rowStart += blockDim.x){
		int rowRead = rowStart + tdx;
		if (rowRead < d && gtdy < d){
			bm[rowRead][tdy] = d_m[rowRead * d + gtdy];
//			printf("\t rowRead, gtdy %d, %d, %f \n",  rowRead, gtdy, bm[rowRead][tdy]);
		}
	}
	if (tdx == 0 && tdy == 0){

	}
	__syncthreads();

	//linear thread block space
	int tid = threadIdx.x * blockDim.y + threadIdx.y;
	int roundEnd = ceilf((float)(d*d)/(blockDim.x * blockDim.y));
	int widthEnd = d % BLOCK_WIDTH;
	for (int r = 0; r < roundEnd; r++){
		float sum = 0;
		int i = (r * blockDim.x * blockDim.y + tid) / d;
		int j = (r * blockDim.x * blockDim.y + tid) % d;
		if (i < d && j < d){
			if (bdy == gridDim.y - 1){
				for (int k = 0; k < widthEnd; k++){
					sum += bm[i][k] * bm[j][k];
//					printf("\t oo sum = ooo gtdy?? -> (%d, %d, %d), (%f, %f)\n",  i, j, k, bm[i][k], bm[j][k]);
				}
//				printf("\t oo sum = %f <- (%d, %d)\n", sum, i, j);
			}
			else{
				for (int k = 0; k < BLOCK_WIDTH; k++){
//					printf("\t oopp sum = ooo gtdy?? -> (%d, %d, %d), (%f, %f)\n",  i, j, k, bm[i][k], bm[j][k]);
					sum += bm[i][k] * bm[j][k];
				}
			}
			atomicAdd(&d_mr[i*d + j], sum);
		}
	}
}
__global__ void cuda_MxMT_v009(float *d_mr, float *d_m, int d){
	__shared__ float bm[BLOCK_WIDTH][1024];
	int bdx = blockIdx.x, bdy = blockIdx.y;
	int tdx = threadIdx.x, tdy = threadIdx.y;
	int gtdx = bdx * blockDim.x + tdx;
	int gtdy = bdy * blockDim.y + tdy;

	for (int rowStart = 0; rowStart < d; rowStart += blockDim.x){
		int rowRead = rowStart + tdx;
		if (rowRead < d && gtdy < d){
			bm[tdy][rowRead] = d_m[rowRead * d + gtdy];
//			printf("\t rowRead, gtdy %d, %d, %f \n",  rowRead, gtdy, bm[rowRead][tdy]);
		}
	}
	if (tdx == 0 && tdy == 0){

	}
	__syncthreads();

	//linear thread block space
	int tid = threadIdx.x * blockDim.y + threadIdx.y;
	int roundEnd = ceilf((float)(d*d)/(blockDim.x * blockDim.y));
	int widthEnd = d % BLOCK_WIDTH;
	for (int r = 0; r < roundEnd; r++){
		float sum = 0;
		int i = (r * blockDim.x * blockDim.y + tid) / d;
		int j = (r * blockDim.x * blockDim.y + tid) % d;
		if (i < d && j < d){
			if (bdy == gridDim.y - 1){
				for (int k = 0; k < widthEnd; k++){
					sum += bm[k][i] * bm[k][j];
//					printf("\t oo sum = ooo gtdy?? -> (%d, %d, %d), (%f, %f)\n",  i, j, k, bm[i][k], bm[j][k]);
				}
//				printf("\t oo sum = %f <- (%d, %d)\n", sum, i, j);
			}
			else{
				for (int k = 0; k < BLOCK_WIDTH; k++){
//					printf("\t oopp sum = ooo gtdy?? -> (%d, %d, %d), (%f, %f)\n",  i, j, k, bm[i][k], bm[j][k]);
					sum += bm[k][i] * bm[k][j];
				}
			}
			atomicAdd(&d_mr[i*d + j], sum);
		}
	}
}
