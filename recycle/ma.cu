#include <iostream>
#include <iomanip>
#include <cstdio>
using namespace std;

const int d = 8;
const int w = 4;

template <class T>
__global__  void runMaxtrix(T *d_m, T *d_mout, int d){
	__shared__ T b_mr[w][w];
	__shared__ T b_mc[w][w];
	int bdx = blockIdx.x;
	int bdy = blockIdx.y;
	int tdx = threadIdx.x;
	int tdy = threadIdx.y;
	int row = bdx * blockDim.x + tdx;
	int col = bdy * blockDim.y + tdy;
	int pos = row*d+col;
	//printf("(%d, %d) (%d, %d): (%d, %d) %d\n", bdx, bdy, tdx, tdy, row, col, d_m[pos]);
	T mval = 0;
	for (int bk = 0; bk < d/w; bk++){
		b_mr[tdx][tdy] = d_m[(bdx*w + tdx)*d + bk * w + tdy]; // (bdx*w+tdx, bk*w+tdy)
		b_mc[tdx][tdy] = d_m[(bdy*w + tdx)*d + bk * w + tdy];	// (bdy*w+tdx, bk*w+tdy)
//		printf("(%d, %d)\n", b_mr[tdx][tdy], b_mc[tdx][tdy]);
		__syncthreads();
		for (int tk = 0; tk < w; tk++){
			mval +=  b_mr[tdx][tk] * b_mc[tdy][tk];
			printf("(%d, %d), tval=->%d\n", row, col, mval);
		}
		__syncthreads();
	}
	d_mout[pos] = mval;
}

template <class T> void diplayMatrix(T *m, int d){
	cout << "m={" << endl;
	for (int i = 0; i < d; i++){
		cout << "\t{";
		for (int j = 0; j < d - 1; j++){
			cout << setw(5) << m[i*d+j] << ", ";
		}
		cout << setw(5) << m[i*d + d - 1] <<
				((i==d-1)? "}" : "},") << endl;
	}
	cout << "};" << endl;
}

int main(int argc, char *argv[]){
	int h_m[d*d];
	int h_mout[d*d];
	for (int i = 0; i < d; i++){
		for (int j = 0; j < d; j++){
			h_m[i*d + j] = i * 100 + j;
		}
	}
	diplayMatrix(h_m, d);
	int *d_m, *d_mout;
	cudaMalloc((void **)&d_m, d*d*sizeof(int));
	cudaMalloc((void **)&d_mout, d*d*sizeof(int));

	cudaMemcpy(d_m, h_m, d*d*sizeof(int), cudaMemcpyHostToDevice);
	runMaxtrix<<<dim3(2,2,1), dim3(4,4,1)>>>(d_m, d_mout, d);
	cudaDeviceSynchronize();
	cudaMemcpy(h_mout, d_mout, d*d*sizeof(int), cudaMemcpyDeviceToHost);
	diplayMatrix(h_mout, d);
	cudaFree(d_m);
	cudaFree(d_mout);
	return 0;
}
