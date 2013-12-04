#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>

#include "seqMatrix.h"
using namespace std;

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

void matrixPopulate(float *m, int d, bool fromRand){
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

